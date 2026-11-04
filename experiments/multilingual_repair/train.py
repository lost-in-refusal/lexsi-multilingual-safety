"""Train a pinned public-data LoRA or reference-preserving SafeGrad adapter.

Every optimizer update uses the frozen train partition. Alignment uses independent
prompt-only inputs and KL(reference || student), summed over positions at batch 1.
This is an explicitly limited protocol, not the paper's aligned-response protocol.
A --max-steps run is feasibility evidence only and never a complete experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'CohereLabs/aya-expanse-8b'
REVISION = '5062468bf9bc0c6035fd64e06274333ec127d980'
SAFETUNE_REVISION = '26f59e6d1bdcacafe5f2be3b8d982fb214369659'
TARGETS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
REQUIRED_FILES = {'config.json', 'train.jsonl', 'calibration.jsonl', 'development.jsonl',
                  'confirmation.jsonl', 'alignment.jsonl'}
PACKAGES = {'torch': '2.15.0', 'transformers': '5.22.1', 'accelerate': '1.15.0',
            'peft': '0.21.2', 'trl': '1.14.2', 'huggingface-hub': '2.4.0'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def validate_config(config):
    require(isinstance(config, dict), 'Configuration must be an object')
    for name, expected in {'model': MODEL, 'model_revision': REVISION, 'seed': 42,
                           'dtype': 'bfloat16', 'device': 'mps',
                           'attention_implementation': 'eager'}.items():
        require(type(config.get(name)) is type(expected) and config[name] == expected,
                f'Unsupported fixed setting: {name}')
    training = config.get('training', {})
    expected = {'epochs': 1, 'learning_rate': 1e-5, 'weight_decay': 0.0, 'batch_size': 1,
                'max_grad_norm': 1.0, 'lora_r': 4, 'lora_alpha': 8, 'lora_dropout': 0.0,
                'max_sequence_tokens': 1024, 'mps_memory_fraction': 0.8,
                'use_reentrant': False, 'rho': 1.0,
                'alignment': 'prompt_only_reference_kl_batchmean_global_projection_plus_rho',
                'reference': 'same_frozen_base_with_adapters_disabled'}
    require(isinstance(training, dict), 'Training configuration must be an object')
    for name, value in expected.items():
        require(type(training.get(name)) is type(value) and training[name] == value,
                f'Unsupported fixed training setting: {name}')
    targets = training.get('target_modules')
    require(isinstance(targets, list) and len(targets) == len(TARGETS)
            and all(isinstance(x, str) for x in targets) and set(targets) == TARGETS,
            'Expected all seven LoRA target modules exactly once')


def validate_token_row(row, *, max_tokens, eos_token_id, alignment=False):
    identity = row.get('id', '<missing>')
    ids, mask = row.get('input_ids'), row.get('attention_mask')
    n, prefix = row.get('sequence_tokens'), row.get('prompt_tokens')
    require(type(n) is int and type(prefix) is int and 0 < prefix <= n <= max_tokens,
            f'{identity}: invalid/oversized token counts; truncation is forbidden')
    require(isinstance(ids, list) and len(ids) == n and all(type(x) is int and x >= 0 for x in ids),
            f'{identity}: invalid input IDs')
    require(isinstance(mask, list) and all(type(x) is int for x in mask) and mask == [1] * n,
            f'{identity}: invalid attention mask')
    if alignment:
        require(n == prefix and 'labels' not in row and 'reference' not in row,
                f'{identity}: alignment must be prompt-only')
    else:
        labels = row.get('labels')
        require(prefix < n and type(eos_token_id) is int and ids[-1] == eos_token_id,
                f'{identity}: missing assistant target or terminal EOS')
        require(isinstance(labels, list) and all(type(x) is int for x in labels)
                and labels == [-100] * prefix + ids[prefix:], f'{identity}: incorrect assistant-only labels')
        require(type(row.get('supervised_tokens')) is int
                and row['supervised_tokens'] == n - prefix == sum(x != -100 for x in labels[1:])
                and row['supervised_tokens'] >= 2, f'{identity}: incorrect supervised-token count')


def load_prepared(directory):
    """Hash every listed basename before parsing train/calibration/alignment only."""
    directory = Path(directory)
    manifest_bytes = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get('status') == 'passed_tokenization_preflight' and manifest.get('truncated_examples') == 0,
            'Prepared bundle must pass preflight with zero truncation')
    files = manifest.get('files')
    require(isinstance(files, dict) and bool(files), 'Missing prepared file hashes')
    blobs = {}
    for name, digest in sorted(files.items()):
        require(isinstance(name, str) and name not in ('', '.', '..', 'manifest.json')
                and Path(name).name == name and '/' not in name and '\\' not in name,
                'Prepared file entries must be safe basenames')
        path = directory / name
        require(not path.is_symlink() and path.is_file(), f'Prepared file missing or symlinked: {name}')
        require(isinstance(digest, str) and len(digest) == 64, f'Invalid digest: {name}')
        content = path.read_bytes()
        require(sha_bytes(content) == digest, f'Prepared file hash mismatch: {name}')
        blobs[name] = content
    require(REQUIRED_FILES <= files.keys(), 'Required prepared files missing')
    config = json.loads(blobs['config.json']); validate_config(config)
    tokenizer = manifest.get('tokenizer', {})
    require(tokenizer.get('actual_snapshot_commit') == REVISION, 'Prepared tokenizer revision mismatch')
    require(type(tokenizer.get('eos_token_id')) is int, 'Missing prepared tokenizer EOS identity')
    splits, seen = {}, set()
    for split in ('train', 'calibration', 'alignment'):
        rows = [json.loads(line) for line in blobs[f'{split}.jsonl'].splitlines() if line.strip()]
        require(bool(rows), f'Empty {split} partition')
        for row in rows:
            require(isinstance(row.get('id'), str) and row['id'] and row['id'] not in seen,
                    'Duplicate/missing training-side identity')
            seen.add(row['id'])
            require(row.get('split') == split and row.get('language') in ('en', 'ko', 'zh'),
                    f'{row["id"]}: invalid partition/language')
            for key in ('source_id', 'request', 'system'):
                require(isinstance(row.get(key), str) and row[key], f'{row["id"]}: missing {key}')
            if split != 'alignment':
                require(isinstance(row.get('reference'), str) and row['reference'] and row.get('category') == 'task',
                        f'{row["id"]}: training/calibration require public task targets')
            validate_token_row(row, max_tokens=config['training']['max_sequence_tokens'],
                               eos_token_id=tokenizer['eos_token_id'], alignment=split == 'alignment')
        splits[split] = rows
    require(Counter(r['language'] for r in splits['train']) == {'en': 450, 'ko': 100, 'zh': 450},
            'Expected the complete 1000-row training partition')
    require(Counter(r['language'] for r in splits['calibration']) == {'en': 6, 'ko': 6, 'zh': 6},
            'Expected 18 calibration rows')
    require(Counter(r['language'] for r in splits['alignment']) == {'en': 24, 'ko': 24, 'zh': 24},
            'Expected 72 independent alignment prompts')
    return {'manifest': manifest, 'manifest_bytes': manifest_bytes, 'manifest_sha256': sha_bytes(manifest_bytes),
            'config': config, 'blobs': blobs, **splits}


def verify_retokenized(bundle):
    """Rebuild complete task inputs and prompt-only alignment with a pinned tokenizer."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    config = bundle['config']
    snapshot = Path(snapshot_download(config['model'], revision=config['model_revision'], local_files_only=True,
                                      allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja']))
    require(snapshot.name == REVISION, 'Cached model snapshot revision mismatch')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    provenance = {'model': MODEL, 'requested_revision': REVISION, 'actual_snapshot_commit': snapshot.name,
                  'eos_token_id': tokenizer.eos_token_id, 'class': type(tokenizer).__name__,
                  'chat_template_sha256': sha_bytes(tokenizer.get_chat_template().encode()),
                  'local_files_only': True, 'trust_remote_code': False}
    for key in ('actual_snapshot_commit', 'eos_token_id', 'chat_template_sha256'):
        require(bundle['manifest']['tokenizer'].get(key) == provenance[key], f'Tokenizer identity differs: {key}')
    for split in ('train', 'calibration', 'alignment'):
        for row in bundle[split]:
            messages = [{'role': 'system', 'content': row['system']}, {'role': 'user', 'content': row['request']}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
            require(prompt == row['input_ids'][:row['prompt_tokens']], f'{row["id"]}: exact prompt mismatch')
            full = prompt
            if split != 'alignment':
                full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': row['reference']}],
                                                      tokenize=True, add_generation_prompt=False, return_dict=False)
                require(tokenizer.decode(full[len(prompt):-1], skip_special_tokens=False) == row['reference'],
                        f'{row["id"]}: target does not round-trip exactly')
            require(full == row['input_ids'], f'{row["id"]}: exact sequence mismatch')
    return tokenizer, snapshot, provenance


def training_order(records, *, seed, max_steps=None):
    require(type(seed) is int and isinstance(records, list) and bool(records), 'Invalid training order inputs')
    require(all(r.get('split') == 'train' for r in records), 'Only train rows may enter optimizer order')
    order = [r.get('id') for r in records]
    require(all(isinstance(i, str) and i for i in order) and len(order) == len(set(order)), 'Invalid/duplicate training IDs')
    random.Random(seed).shuffle(order)
    if max_steps is not None:
        require(type(max_steps) is int and 0 < max_steps < len(order),
                'Feasibility --max-steps must be positive and smaller than a full epoch')
        order = order[:max_steps]
    return order


def combine_gradients(task, alignment, *, rho):
    """Global surgery followed by +rho alignment; exact zero alignment is valid."""
    import torch
    require(type(rho) in (int, float) and math.isfinite(rho) and rho >= 0, 'Invalid alignment weight')
    require(len(task) == len(alignment) and len(task) > 0, 'Gradient coverage mismatch')
    dots, task_sq, align_sq = [], [], []
    for user, align in zip(task, alignment):
        require(isinstance(user, torch.Tensor) and isinstance(align, torch.Tensor), 'Missing task/alignment gradient')
        require(user.shape == align.shape and user.device == align.device, 'Gradient shape/device mismatch')
        require(torch.isfinite(user).all().item() and torch.isfinite(align).all().item(), 'Nonfinite task/alignment gradient')
        dots.append((user.float() * align.float()).sum().item())
        task_sq.append(user.float().square().sum().item()); align_sq.append(align.float().square().sum().item())
    dot, user_sq, safety_sq = math.fsum(dots), math.fsum(task_sq), math.fsum(align_sq)
    require(all(math.isfinite(x) for x in (dot, user_sq, safety_sq)), 'Nonfinite global gradient statistic')
    conflict = dot < 0 and safety_sq > 0
    coefficient = dot / safety_sq if conflict else 0.0
    result = [user + (rho - coefficient) * align for user, align in zip(task, alignment)]
    require(all(torch.isfinite(x).all().item() for x in result), 'Nonfinite combined gradient')
    return result, {'dot': dot, 'task_norm': math.sqrt(user_sq), 'alignment_norm': math.sqrt(safety_sq),
                    'conflict': conflict, 'zero_alignment': safety_sq == 0,
                    'projection_coefficient': coefficient, 'rho': float(rho),
                    'cosine': dot / math.sqrt(user_sq * safety_sq) if user_sq > 0 and safety_sq > 0 else None}


@contextmanager
def model_mode(model, *, training, checkpointing):
    modes = [(module, module.training) for module in model.modules()]
    old_checkpointing = bool(getattr(model, 'is_gradient_checkpointing', False))
    try:
        model.train(training)
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            model.gradient_checkpointing_disable()
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        if old_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            model.gradient_checkpointing_disable()


@contextmanager
def disabled_adapter_reference(model):
    """PEFT temporarily mutates requires_grad; restore every original flag on exit."""
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        with model_mode(model, training=False, checkpointing=False), model.disable_adapter():
            yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


def chunked_alignment_loss(student_logits, teacher_logits, kl_helper, *, chunk_tokens=32):
    """Keep float32 KL workspaces bounded; checkpoint each exact pinned KL chunk."""
    import torch
    from torch.utils.checkpoint import checkpoint
    require(student_logits.shape == teacher_logits.shape and student_logits.ndim == 3,
            'Alignment logit shapes differ')
    require(student_logits.shape[0] == 1 and student_logits.shape[1] > 0, 'Alignment requires batch size one')
    require(type(chunk_tokens) is int and chunk_tokens > 0, 'Invalid KL chunk size')
    total = student_logits.new_zeros((), dtype=torch.float32)
    for offset in range(0, student_logits.shape[1], chunk_tokens):
        student = student_logits[:, offset:offset + chunk_tokens]
        teacher = teacher_logits[:, offset:offset + chunk_tokens].detach()
        def objective(s, t):
            return kl_helper(s.float(), t.float(), temperature=1.0, reduction='batchmean')
        total = total + checkpoint(objective, student, teacher, use_reentrant=False)
    require(torch.isfinite(total).item(), 'Nonfinite alignment loss')
    return total


def verify_attention_backend(model, expected):
    """Inspect the effective model and every decoder attention configuration."""
    require(expected == 'eager', 'Unsupported fixed attention backend')
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    effective = getattr(base.config, '_attn_implementation', None)
    attention = [module for name, module in base.named_modules() if name.endswith('self_attn')]
    require(effective == expected and bool(attention)
            and all(getattr(module.config, '_attn_implementation', None) == expected for module in attention),
            'Effective attention backend differs from the fixed eager configuration')
    return {'requested': expected, 'effective': effective, 'attention_modules': len(attention)}


def pin_attention_backend(model, requested):
    """Pin attention before any forward; SafeTune's loader has no backend kwarg."""
    require(requested == 'eager', 'Unsupported fixed attention backend')
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    base.set_attn_implementation(requested)
    return verify_attention_backend(model, requested)


def alignment_gradients(model, batch, parameters, kl_helper, *, chunk_tokens=32, memory=None,
                        require_initial_equivalence=False, diagnostics=None):
    import torch
    from torch.utils.checkpoint import checkpoint
    diagnostic = diagnostics if diagnostics is not None else {}
    require(torch.is_grad_enabled(), 'Alignment requires an autograd-enabled student pass')
    if require_initial_equivalence:
        diagnostic['status'] = 'failed'
        initial_b = [p for name, p in model.named_parameters() if '.lora_B.' in name]
        diagnostic['initial_lora_B_zero'] = bool(initial_b) and all(torch.count_nonzero(p.detach()).item() == 0
                                                                  for p in initial_b)
        require(diagnostic['initial_lora_B_zero'], 'Initial reference check requires exactly zero LoRA B tensors')
    with disabled_adapter_reference(model), torch.no_grad():
        teacher = model(**batch, use_cache=False).logits.detach()
    require(not teacher.requires_grad and torch.isfinite(teacher).all().item(), 'Invalid reference logits')
    if memory:
        memory('alignment_reference')
    # Transformer built-in checkpointing skips eval mode. This outer nonreentrant
    # checkpoint preserves eval semantics and recomputes the student grad pass.
    with model_mode(model, training=False, checkpointing=False):
        def student_forward(ids, mask):
            return model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        student = checkpoint(student_forward, batch['input_ids'], batch['attention_mask'], use_reentrant=False)
        diagnostic['student_requires_grad'] = student.requires_grad
        require(student.requires_grad, 'Alignment student logits must require gradients')
        require(torch.isfinite(student).all().item(), 'Nonfinite alignment student logits')
        equal = torch.equal(student.detach(), teacher) and student.dtype == teacher.dtype
        if require_initial_equivalence:
            diagnostic['reference_logits_equal'] = equal
            diagnostic['max_abs_logit_difference'] = (student.detach().float() - teacher.float()).abs().max().item()
            require(equal, 'Exact initial reference/student logits differ on the autograd-enabled path')
        loss = chunked_alignment_loss(student, teacher, kl_helper, chunk_tokens=chunk_tokens)
        if require_initial_equivalence:
            diagnostic['raw_pinned_helper_kl'] = loss.detach().item()
        # The pinned helper can have a roundoff residual even at identical
        # logits. Enforce the analytic stationary point only for exact equality;
        # otherwise a tiny numerical residual can trigger a large projection.
        # Compute every gradient through the checked objective; never substitute
        # for missing or nonfinite gradients, and never apply a tolerance here.
        if equal:
            loss = loss * 0.0
        if memory:
            memory('alignment_forward')
        gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    require(all(torch.isfinite(g).all().item() for g in gradients), 'Nonfinite alignment gradients')
    value = loss.detach().item()
    if require_initial_equivalence:
        diagnostic['initial_alignment_loss'] = value
        diagnostic['initial_alignment_gradients_zero'] = all(torch.count_nonzero(g).item() == 0 for g in gradients)
        require(value == 0.0 and diagnostic['initial_alignment_gradients_zero'],
                'Initial alignment loss/gradients must be exactly zero')
        diagnostic['status'] = 'passed'
    if memory:
        memory('alignment_backward')
    return value, [g.detach() for g in gradients]


def task_loss(model, row, *, device, chunk_tokens=32):
    """Exact assistant+EOS next-token CE without full-sequence float32 logits."""
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint
    prefix, n = row['prompt_tokens'], row['sequence_tokens']
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
    mask = torch.tensor([row['attention_mask']], dtype=torch.long, device=device)
    positions = torch.arange(prefix - 1, n - 1, device=device)
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=positions, use_cache=False).logits
    targets = ids[:, prefix:]
    require(logits.shape[:2] == targets.shape, 'Selected task logits do not match assistant targets')
    total = logits.new_zeros((), dtype=torch.float32)
    for offset in range(0, targets.shape[1], chunk_tokens):
        def objective(prediction, target):
            return functional.cross_entropy(prediction.float().reshape(-1, prediction.shape[-1]),
                                            target.reshape(-1), reduction='sum')
        predictions, target = logits[:, offset:offset + chunk_tokens], targets[:, offset:offset + chunk_tokens]
        if torch.is_grad_enabled():
            total = total + checkpoint(objective, predictions, target, use_reentrant=False)
        else:
            total = total + objective(predictions, target)
    loss = total / row['supervised_tokens']
    require(torch.isfinite(loss).item(), 'Nonfinite task loss')
    return loss


def aggregate_calibration(measurements):
    require(bool(measurements), 'Calibration requires measurements')
    require(len({r['id'] for r in measurements}) == len(measurements), 'Duplicate calibration measurement')
    require(all(math.isfinite(r['loss']) and r['loss'] >= 0 and r['supervised_tokens'] > 0 for r in measurements),
            'Invalid calibration measurement')
    count = sum(r['supervised_tokens'] for r in measurements)
    return {'rows': len(measurements), 'supervised_tokens': count,
            'token_weighted_loss': sum(r['loss'] * r['supervised_tokens'] for r in measurements) / count,
            'measurements': measurements, 'scope': 'Forward-only diagnostic; no checkpoint selection.'}


def verify_environment():
    import safetune
    from safetune.runner.utils.model_utils import load_model, lora_wrap
    from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
    source = ROOT / 'vendor/SafeTune'
    require(sys.version_info[:2] == (3, 12), 'Expected Python 3.12')
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(versions == PACKAGES, 'Pinned training package versions differ')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    require(commit == SAFETUNE_REVISION and not dirty, 'SafeTune revision/cleanliness mismatch')
    require(Path(safetune.__file__).resolve().is_relative_to(source.resolve()), 'Unexpected SafeTune import source')
    hashes = {}
    for function in (load_model, lora_wrap, safegrad_kl_alignment_loss):
        path = Path(inspect.getfile(function)).resolve()
        require(path.is_relative_to(source.resolve()), 'Unexpected SafeTune implementation source')
        hashes[str(path.relative_to(source.resolve()))] = sha_bytes(path.read_bytes())
    return {'SafeTune': commit, 'packages': versions, 'source_hashes': hashes,
            'python': '.'.join(map(str, sys.version_info[:3]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('plain', 'safegrad'), required=True)
    parser.add_argument('--max-steps', type=int, help='Distinctly labeled feasibility prefix, smaller than a complete epoch')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; use a new directory')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {'status': 'running', 'stage': 'verify_prepared', 'arm': args.arm,
              'run_kind': 'feasibility' if args.max_steps is not None else 'full_epoch',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'optimizer_steps': 0,
              'model_weights_loaded': False, 'adapter_directory': 'adapter', 'memory_snapshots': [],
              'scope': 'Public-data adapter experiment; optimization diagnostics do not establish safety or task improvement.',
              'alignment_protocol': 'Independent prompt-only KL(reference||student), temperature1, batchmean; summed positions at batch1.',
              'alignment_stationarity': 'Exactly equal finite reference/student logits use analytic zero KL and zero gradients; no tolerance.',
              'alignment_protocol_limit': 'Different from aligned-response training; no claim of paper-identical protocol.',
              'memory_measurement_scope': 'Synchronized MPS stage samples and process RSS high-water; not a continuous MPS peak.'}

    def save_report():
        write_json(args.output / 'report.json', report)

    def stage(name):
        report['stage'] = name
        save_report()
        print(name, flush=True)

    save_report()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        bundle = load_prepared(args.prepared)
        config, training = bundle['config'], bundle['config']['training']
        order = training_order(bundle['train'], seed=config['seed'], max_steps=args.max_steps)
        report.update(model=config['model'], model_revision=config['model_revision'],
                      attention_implementation=config['attention_implementation'],
                      training_config=config, config_sha256=sha_bytes(bundle['blobs']['config.json']),
                      prepared_manifest_sha256=bundle['manifest_sha256'],
                      prepared_file_hashes=bundle['manifest']['files'],
                      training_script_sha256=sha_bytes(Path(__file__).read_bytes()),
                      training_order=order, expected_steps=len(order), complete_epoch_rows=len(bundle['train']),
                      optimizer={'name': 'AdamW', 'learning_rate': training['learning_rate'],
                                 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'epsilon': 1e-8,
                                 'foreach': False, 'batch_size': 1, 'gradient_accumulation_steps': 1,
                                 'max_gradient_norm': training['max_grad_norm']})
        (args.output / 'training-config.json').write_bytes(bundle['blobs']['config.json'])
        (args.output / 'prepared-manifest.json').write_bytes(bundle['manifest_bytes'])
        (args.output / 'training-source.py').write_bytes(Path(__file__).read_bytes())
        stage('verify_exact_tokenization')
        tokenizer, snapshot, provenance = verify_retokenized(bundle)
        report['tokenizer'] = provenance
        report['exact_retokenization_rows'] = sum(len(bundle[s]) for s in ('train', 'calibration', 'alignment'))
        del tokenizer
        stage('verify_training_environment')
        report['environment'] = verify_environment()
        import torch
        import safetune
        from peft import get_peft_model_state_dict
        from safetune.runner.utils.model_utils import load_model, lora_wrap
        from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
        require(torch.backends.mps.is_available(), 'Configured MPS backend is unavailable')
        torch.mps.set_per_process_memory_fraction(training['mps_memory_fraction'])
        report['mps_recommended_max_gib'] = torch.mps.recommended_max_memory() / 2**30
        report['mps_memory_fraction'] = training['mps_memory_fraction']

        def reset_seed():
            safetune.configure(device='mps', seed=config['seed'])
            random.seed(config['seed']); torch.manual_seed(config['seed']); torch.mps.manual_seed(config['seed'])

        def memory(label):
            torch.mps.synchronize()
            record = {'stage': label, 'elapsed_seconds': round(time.perf_counter() - start, 3),
                      'mps_tensor_gib': torch.mps.current_allocated_memory() / 2**30,
                      'mps_driver_gib': torch.mps.driver_allocated_memory() / 2**30,
                      'process_max_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30}
            report['memory_snapshots'].append(record)
            return record

        reset_seed()
        stage('load_fresh_frozen_base')
        model = load_model(str(snapshot), dtype='bfloat16', device='mps')
        report['model_weights_loaded'] = True
        report['attention_backend'] = pin_attention_backend(model, config['attention_implementation'])
        require(next(model.parameters()).device.type == 'mps' and next(model.parameters()).dtype == torch.bfloat16,
                'Base model device/dtype differs')
        memory('base_loaded')
        model = lora_wrap(model, r=training['lora_r'], lora_alpha=training['lora_alpha'],
                          lora_dropout=training['lora_dropout'], target_modules=training['target_modules'])
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        require(bool(trainable) and all('lora_' in name for name in trainable), 'Unexpected trainable base parameters')
        require(all(p.dtype == torch.float32 for p in trainable.values()), 'Expected FP32 adapter parameters')
        require(set(model.peft_config['default'].target_modules) == TARGETS, 'Actual LoRA target modules differ')
        parameters = list(trainable.values())
        frozen = [(name, p, p.detach().flatten()[:1024].cpu().clone())
                  for name, p in model.named_parameters() if not p.requires_grad]
        initial_adapter = {name: p.detach().cpu().clone() for name, p in trainable.items()}
        report.update(trainable_parameters=sum(p.numel() for p in parameters),
                      trainable_dtypes=sorted({str(p.dtype) for p in parameters}),
                      target_modules=sorted(TARGETS), all_base_parameters_frozen=True,
                      frozen_parameter_samples_checked=len(frozen),
                      frozen_parameter_check_scope='First1024 elements of every frozen tensor; requires_grad and grad checks throughout. Not a full-weight hash.')

        def check_base():
            require(all(not p.requires_grad and p.grad is None for _, p, _ in frozen), 'Frozen base received gradients/trainability')
            require(all(p.requires_grad for p in parameters), 'Adapter trainability was lost')

        def task_grads(row, label):
            with model_mode(model, training=True, checkpointing=True):
                loss = task_loss(model, row, device='mps')
                memory(label + '_task_forward')
                gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
            require(all(torch.isfinite(g).all().item() for g in gradients), 'Nonfinite task gradients')
            value = loss.detach().item()
            memory(label + '_task_backward')
            return value, [g.detach() for g in gradients]

        def align_grads(row, label, *, initial=False, diagnostics=None):
            batch = {key: torch.tensor([row[key]], dtype=torch.long, device='mps')
                     for key in ('input_ids', 'attention_mask')}
            return alignment_gradients(model, batch, parameters, safegrad_kl_alignment_loss,
                                       memory=lambda name: memory(label + '_' + name),
                                       require_initial_equivalence=initial, diagnostics=diagnostics)

        def put_gradients(gradients):
            require(len(gradients) == len(parameters), 'Gradient coverage mismatch')
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            check_base()
            norm = torch.nn.utils.clip_grad_norm_(parameters, training['max_grad_norm'], error_if_nonfinite=True).item()
            require(math.isfinite(norm) and norm > 0, 'Missing/zero/nonfinite combined update')
            return norm

        stage('longest_inputs_memory_preflight')
        longest_task = max(bundle['train'], key=lambda r: r['sequence_tokens'])
        longest_alignment = max(bundle['alignment'], key=lambda r: r['sequence_tokens'])
        preflight_start = time.perf_counter()
        report['attention_backend'] = verify_attention_backend(model, config['attention_implementation'])
        report['initial_reference_preflight'] = {'status': 'running', 'sample_id': longest_alignment['id'],
                                                'optimizer_steps': report['optimizer_steps']}
        alignment_loss, alignment = align_grads(longest_alignment, 'preflight', initial=True,
                                                diagnostics=report['initial_reference_preflight'])
        task_value, task = task_grads(longest_task, 'preflight')
        gradients, stats = (combine_gradients(task, alignment, rho=training['rho'])
                            if args.arm == 'safegrad' else (task, None))
        norm = put_gradients(gradients)
        model.zero_grad(set_to_none=True)
        require(all(torch.equal(initial_adapter[name], p.detach().cpu()) for name, p in trainable.items()),
                'Zero-update memory preflight changed adapter parameters')
        report['memory_preflight'] = {'status': 'passed', 'optimizer_steps': 0, 'adapter_state_unchanged': True,
            'task_id': longest_task['id'], 'task_sequence_tokens': longest_task['sequence_tokens'],
            'alignment_id': longest_alignment['id'],
            'alignment_sequence_tokens': longest_alignment['sequence_tokens'],
            'initial_reference_equivalence_verified': True,
            'task_loss': task_value, 'alignment_loss': alignment_loss, 'gradient_norm_before_clip': norm,
            'gradient_statistics': stats, 'seconds': time.perf_counter() - preflight_start}
        del task, alignment, gradients
        gc.collect(); torch.mps.empty_cache(); reset_seed()
        memory('preflight_cleared')

        calibration_rows = bundle['calibration']
        if args.max_steps is not None:
            calibration_rows = [r for language in ('en', 'ko', 'zh')
                                for r in [x for x in bundle['calibration'] if x['language'] == language][:2]]
        report['calibration_ids'] = [r['id'] for r in calibration_rows]
        report['calibration_scope'] = ('Feasibility subset: first2 prepared calibration rows per language; no selection.'
                                       if args.max_steps is not None else 'All18 frozen calibration rows; no selection.')

        def calibrate(label):
            stage(label)
            measurements = []
            model.zero_grad(set_to_none=True)
            with model_mode(model, training=False, checkpointing=False), torch.no_grad():
                for row in calibration_rows:
                    value = task_loss(model, row, device='mps').item()
                    measurements.append({'id': row['id'], 'language': row['language'], 'loss': value,
                                         'supervised_tokens': row['supervised_tokens']})
            check_base(); memory(label)
            return aggregate_calibration(measurements)

        report['calibration_before'] = calibrate('calibration_before')
        reset_seed()
        align_order = list(bundle['alignment']); random.Random(config['seed'] + 1).shuffle(align_order)
        report['alignment_order'] = ([align_order[i % len(align_order)]['id'] for i in range(len(order))]
                                     if args.arm == 'safegrad' else [])
        by_id = {r['id']: r for r in bundle['train']}
        optimizer = torch.optim.AdamW(parameters, lr=training['learning_rate'], weight_decay=0., foreach=False)
        train_start = time.perf_counter()
        with (args.output / 'steps.jsonl').open('x', encoding='utf-8') as log:
            for number, identity in enumerate(order, 1):
                stage(f'training_step_{number}')
                row = by_id[identity]
                require(row['split'] == 'train', 'Nontraining row entered optimizer')
                optimizer.zero_grad(set_to_none=True)
                step_start = time.perf_counter()
                initial_diagnostic = {} if number == 1 and args.arm == 'safegrad' else None
                safety_value, alignment = (align_grads(align_order[(number - 1) % len(align_order)], f'step{number}',
                                                       initial=number == 1, diagnostics=initial_diagnostic)
                                           if args.arm == 'safegrad' else (None, None))
                value, task = task_grads(row, f'step{number}')
                gradients, stats = (combine_gradients(task, alignment, rho=training['rho'])
                                    if args.arm == 'safegrad' else (task, None))
                norm = put_gradients(gradients)
                optimizer.step()
                report['optimizer_steps'] += 1
                finite = all(torch.isfinite(p).all().item() for p in parameters)
                record = {'step': number, 'sample_id': identity, 'split': 'train', 'language': row['language'],
                          'task_loss': value, 'supervised_tokens': row['supervised_tokens'],
                          'alignment_id': report['alignment_order'][number - 1] if args.arm == 'safegrad' else None,
                          'alignment_loss': safety_value, 'gradient_statistics': stats,
                          'initial_reference_diagnostic': initial_diagnostic,
                          'gradient_norm_before_clip': norm, 'finite_adapter_weights': finite,
                          'seconds': time.perf_counter() - step_start}
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                log.flush(); os.fsync(log.fileno())
                require(finite, 'Nonfinite adapter weights after update')
                check_base(); memory(f'step{number}_optimizer')
                report['last_completed_step'] = record
                save_report()
                del task, alignment, gradients
        report['training_seconds'] = time.perf_counter() - train_start
        require(report['optimizer_steps'] == report['expected_steps'], 'Incomplete fixed training order')
        logged = [json.loads(line) for line in (args.output / 'steps.jsonl').read_text().splitlines()]
        require([row['sample_id'] for row in logged] == order, 'Actual optimizer order differs from frozen order')
        changes = [(p.detach().cpu() - initial_adapter[name]).abs().max().item() for name, p in trainable.items()]
        report.update(changed_adapter_tensors=sum(x > 0 for x in changes), maximum_adapter_absolute_change=max(changes))
        require(report['changed_adapter_tensors'] > 0, 'No adapter tensor changed')
        report['frozen_parameter_samples_unchanged'] = all(torch.equal(saved, p.detach().flatten()[:1024].cpu())
                                                          for _, p, saved in frozen)
        require(report['frozen_parameter_samples_unchanged'], 'Frozen base sample changed')
        check_base(); optimizer.zero_grad(set_to_none=True)
        del optimizer, initial_adapter
        gc.collect(); torch.mps.empty_cache()
        report['calibration_after'] = calibrate('calibration_after')
        stage('save_and_reload_adapter')
        adapter = args.output / 'adapter'
        model.peft_config['default'].base_model_name_or_path = MODEL
        model.peft_config['default'].revision = REVISION
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        original = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
        require(bool(original) and all('lora_' in key and torch.isfinite(value).all().item() for key, value in original.items()),
                'Saved state must contain finite adapter tensors only')
        model.load_adapter(adapter, adapter_name='reloaded', is_trainable=False)
        loaded = get_peft_model_state_dict(model, adapter_name='reloaded')
        report['saved_adapter_reload_exact'] = (original.keys() == loaded.keys()
            and all(torch.equal(value, loaded[key].detach().cpu()) for key, value in original.items()))
        require(report['saved_adapter_reload_exact'], 'Reloaded adapter tensors differ')
        report['adapter_files'] = {str(path.relative_to(adapter)): sha_bytes(path.read_bytes())
                                  for path in sorted(adapter.rglob('*')) if path.is_file()}
        require({'adapter_config.json', 'adapter_model.safetensors'} <= report['adapter_files'].keys(), 'Missing adapter artifacts')
        report['adapter_weights_bytes'] = (adapter / 'adapter_model.safetensors').stat().st_size
        report['steps_sha256'] = sha_bytes((args.output / 'steps.jsonl').read_bytes())
        report['maximum_observed_stage_mps_driver_gib'] = max(r['mps_driver_gib'] for r in report['memory_snapshots'])
        report.update(status='passed_feasibility' if args.max_steps is not None else 'passed_training', stage='complete')
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error_type=type(exc).__name__, error='Details retained in local error.log')
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.perf_counter() - start
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_report()
        print(report['status'], flush=True)
    return 0 if report['status'] in ('passed_training', 'passed_feasibility') else 1


if __name__ == '__main__':
    raise SystemExit(main())
