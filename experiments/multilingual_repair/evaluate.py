#!/usr/bin/env python3
"""Generate frozen development/alignment outputs; semantic review is separate.

Task rows contain reference-answer tokens for training. Generation uses only the
verified prompt prefix, never the reference suffix. Confirmation is deliberately
unavailable until an independently reviewed selection protocol is implemented.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'CohereLabs/aya-expanse-8b'
MODEL_REVISION = '5062468bf9bc0c6035fd64e06274333ec127d980'
TARGET_MODULES = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
PREPARED_FILES = {'config.json', 'train.jsonl', 'calibration.jsonl', 'development.jsonl',
                  'confirmation.jsonl', 'alignment.jsonl'}
TOOL_REVISIONS = {'SafeTune': '26f59e6d1bdcacafe5f2be3b8d982fb214369659',
                  'AuditKIT': '71316da345784ff220d5ed59337441a59ab31526'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write_json(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_bytes(json_bytes(value))
    pending.replace(path)


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def validate_config(config):
    require(isinstance(config, dict), 'Configuration must be an object')
    require(config.get('model') == MODEL and config.get('model_revision') == MODEL_REVISION,
            'Expected the pinned Aya Expanse model and immutable revision')
    require(config.get('dtype') == 'bfloat16' and config.get('device') == 'mps',
            'Expected the frozen bfloat16 MPS runtime')
    require(config.get('attention_implementation') == 'eager',
            'The frozen attention implementation must be eager')
    require(type(config.get('seed')) is int and config['seed'] == 42, 'Expected fixed seed 42')
    generation = config.get('generation', {})
    require(type(generation.get('temperature')) in (int, float) and generation['temperature'] == 0,
            'Only deterministic greedy generation with temperature zero is supported')
    require(type(generation.get('max_new_tokens')) is int and generation['max_new_tokens'] > 0,
            'max_new_tokens must be a positive integer')
    require(generation.get('batch_size', 1) == 1, 'Generation requires batch size one')
    training = config.get('training', {})
    require(training.get('batch_size', 1) == 1, 'Training requires batch size one')
    require(type(training.get('max_sequence_tokens')) is int and training['max_sequence_tokens'] > 0,
            'Missing immutable maximum sequence length')
    require(type(training.get('epochs')) is int and training['epochs'] > 0, 'Invalid epoch count')
    require(type(training.get('mps_memory_fraction')) in (float, int)
            and 0 < training['mps_memory_fraction'] <= 1, 'Invalid MPS memory fraction')
    return config


def prompt_inputs(row):
    """Select the preserved prompt prefix; full task rows include a reference."""
    ids, mask, prefix = row.get('input_ids'), row.get('attention_mask'), row.get('prompt_tokens')
    require(isinstance(ids, list) and bool(ids) and all(type(i) is int and i >= 0 for i in ids),
            'Invalid prepared input token IDs')
    require(type(prefix) is int and 0 < prefix <= len(ids), 'Missing or invalid prompt boundary')
    require(isinstance(mask, list) and len(mask) == len(ids)
            and all(type(i) is int and i == 1 for i in mask), 'Invalid prepared attention mask')
    require(row.get('sequence_tokens') == len(ids), 'Prepared sequence token count differs')
    if row.get('category') == 'task':
        require(isinstance(row.get('reference'), str) and bool(row['reference'].strip()),
                'Task row requires a reference for separate semantic review')
        require(prefix < len(ids) and row.get('labels') == [-100] * prefix + ids[prefix:],
                'Task row has an invalid assistant-only label boundary')
    else:
        require(prefix == len(ids), 'Prompt-only row contains an unexpected reference suffix')
    return ids[:prefix], mask[:prefix]


def load_prepared(directory, split='development'):
    """Verify all archive digests, but never parse held-out confirmation rows."""
    require(split in ('development', 'alignment'),
            'Only development/alignment generation is enabled; confirmation requires a later independent protocol')
    directory = Path(directory)
    manifest_data = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_data)
    require(isinstance(manifest, dict), 'Manifest must be an object')
    require(manifest.get('status') == 'passed_tokenization_preflight'
            and manifest.get('truncated_examples') == 0, 'Preparation must pass preflight with zero truncation')
    files = manifest.get('files')
    require(isinstance(files, dict) and set(files) == PREPARED_FILES,
            'Unexpected prepared file names; only the fixed archive basenames are accepted')
    blobs = {}
    for name, expected in files.items():
        require(is_digest(expected), 'Invalid prepared file digest')
        data = (directory / name).read_bytes()
        require(sha256(data) == expected, f'Prepared file hash mismatch: {name}')
        # Confirmation bytes are hashed as an opaque integrity check only.
        if name != 'confirmation.jsonl':
            blobs[name] = data
    config = validate_config(json.loads(blobs['config.json']))
    tokenizer = manifest.get('tokenizer', {})
    require(tokenizer.get('actual_snapshot_commit') == config['model_revision'], 'Prepared tokenizer revision mismatch')
    require(type(tokenizer.get('eos_token_id')) is int and tokenizer['eos_token_id'] >= 0,
            'Prepared tokenizer EOS is missing')
    require(is_digest(tokenizer.get('chat_template_sha256')), 'Prepared tokenizer template digest is missing')
    rows = [json.loads(line) for line in blobs[split + '.jsonl'].splitlines() if line.strip()]
    require(bool(rows), 'Selected evaluation split is empty')
    seen = set()
    for row in rows:
        require(isinstance(row, dict), 'Prepared evaluation row must be an object')
        require(isinstance(row.get('id'), str) and row['id'] and row['id'] not in seen,
                'Duplicate or missing evaluation ID')
        seen.add(row['id'])
        require(row.get('split') == split, 'Evaluation row belongs to another split')
        require(row.get('language') in ('en', 'ko', 'zh'), 'Unexpected evaluation language')
        require(isinstance(row.get('source_id'), str) and bool(row['source_id']), 'Missing public source ID')
        require(isinstance(row.get('request'), str) and bool(row['request'].strip())
                and isinstance(row.get('system'), str), 'Missing original request/system text')
        require(row.get('category') in ('task', 'disallowed'), 'Unexpected evaluation category')
        prompt_inputs(row)
        require(len(row['input_ids']) <= config['training']['max_sequence_tokens'],
                'Prepared row exceeds the frozen sequence cap; truncation is forbidden')
    train_rows = [json.loads(line) for line in blobs['train.jsonl'].splitlines() if line.strip()]
    train_ids = [row.get('id') for row in train_rows]
    require(bool(train_ids) and all(isinstance(i, str) and i for i in train_ids)
            and len(set(train_ids)) == len(train_ids), 'Invalid prepared training IDs')
    return {'manifest': manifest, 'manifest_sha256': sha256(manifest_data), 'config': config,
            'config_sha256': files['config.json'], 'config_bytes': blobs['config.json'],
            'rows': rows, 'train_ids': train_ids, 'split': split}


def load_training_run(directory, bundle):
    """Validate completed adapter-only training and every actual adapter byte."""
    directory = Path(directory)
    report_data = (directory / 'report.json').read_bytes()
    config_data = (directory / 'training-config.json').read_bytes()
    manifest_data = (directory / 'prepared-manifest.json').read_bytes()
    report, config = json.loads(report_data), json.loads(config_data)
    require(isinstance(report, dict), 'Training report must be an object')
    feasible_training = report.get('status') == 'passed_feasibility'
    require(report.get('status') in ('passed_training', 'passed_feasibility'),
            'A completed passed_training or passed_feasibility run is required')
    require(report.get('run_kind') == ('feasibility' if feasible_training else 'full_epoch'),
            'Training run kind/status mismatch')
    require(report.get('arm') in ('plain', 'safegrad'), 'Unknown training arm')
    require(sha256(config_data) == report.get('config_sha256') == bundle['config_sha256']
            and config == report.get('training_config') == bundle['config'], 'Training configuration/hash mismatch')
    require(sha256(manifest_data) == report.get('prepared_manifest_sha256') == bundle['manifest_sha256'],
            'Training prepared manifest hash mismatch')
    require(report.get('prepared_file_hashes') == bundle['manifest']['files'], 'Training prepared file hashes differ')
    for field in ('model', 'model_revision'):
        require(report.get(field) == config[field], f'Training base {field} mismatch')
    require(report.get('attention_implementation') == config['attention_implementation'] == 'eager',
            'Training attention implementation differs from the frozen eager backend')
    require(is_digest(report.get('training_script_sha256')), 'Training source digest is missing')
    expected = report.get('expected_steps')
    require(type(expected) is int and expected > 0 and type(report.get('optimizer_steps')) is int
            and report['optimizer_steps'] == expected, 'Incomplete training optimizer steps')
    total = len(bundle['train_ids']) * config['training']['epochs']
    order = report.get('training_order')
    require(isinstance(order, list) and len(order) == expected and set(order) <= set(bundle['train_ids']),
            'Training update order is incomplete or includes unapproved IDs')
    if feasible_training:
        require(expected <= total, 'Feasibility schedule exceeds the complete epoch schedule')
    else:
        require(expected == total and Counter(order) == Counter({i: config['training']['epochs'] for i in bundle['train_ids']}),
                'Training did not complete every frozen epoch')
    for field in ('saved_adapter_reload_exact', 'all_base_parameters_frozen', 'frozen_parameter_samples_unchanged'):
        require(report.get(field) is True, f'Missing completed training check: {field}')
    require(report.get('adapter_directory') == 'adapter', 'Expected adapter-only relative directory')
    adapter = directory / 'adapter'
    files = report.get('adapter_files')
    require(isinstance(files, dict) and {'adapter_config.json', 'adapter_model.safetensors'} <= set(files)
            and set(files) <= {'adapter_config.json', 'adapter_model.safetensors', 'README.md'}
            and all(is_digest(v) for v in files.values()), 'Invalid adapter file manifest')
    actual = {str(p.relative_to(adapter)) for p in adapter.rglob('*') if p.is_file()}
    require(actual == set(files), 'Adapter directory files differ from training manifest')
    for name, expected_digest in files.items():
        require(sha256((adapter / name).read_bytes()) == expected_digest, f'Adapter file hash mismatch: {name}')
    adapter_config = json.loads((adapter / 'adapter_config.json').read_bytes())
    training = config['training']
    for field, expected_value in (
        ('base_model_name_or_path', config['model']), ('revision', config['model_revision']),
        ('peft_type', 'LORA'), ('task_type', 'CAUSAL_LM'), ('r', training['lora_r']),
        ('lora_alpha', training['lora_alpha']), ('lora_dropout', training['lora_dropout']),
        ('bias', 'none'), ('inference_mode', True)):
        require(adapter_config.get(field) == expected_value, f'Saved adapter configuration mismatch: {field}')
    require(set(adapter_config.get('target_modules', [])) == set(training['target_modules']) == TARGET_MODULES,
            'Saved adapter target modules mismatch')
    for field in ('modules_to_save', 'rank_pattern', 'alpha_pattern', 'use_dora', 'use_rslora', 'lora_bias',
                  'trainable_token_indices', 'target_parameters', 'layer_replication', 'fan_in_fan_out',
                  'exclude_modules', 'layers_to_transform', 'layers_pattern', 'use_qalora', 'use_bdlora',
                  'alora_invocation_tokens'):
        require(not adapter_config.get(field), f'Unapproved saved adapter feature: {field}')
    identity = {key: report[key] for key in ('arm', 'run_kind', 'model', 'model_revision', 'attention_implementation', 'config_sha256',
                'prepared_manifest_sha256', 'training_script_sha256', 'expected_steps', 'optimizer_steps', 'adapter_files')}
    identity.update(training_report_sha256=sha256(report_data), feasibility_only=feasible_training)
    return {'adapter_path': adapter, 'identity': identity}


def verify_attention(model, config):
    """Reject a model that did not retain the explicitly selected eager backend."""
    expected = config.get('attention_implementation')
    require(expected == 'eager', 'The frozen attention implementation must be eager')
    require(getattr(model.config, '_attn_implementation', None) == expected,
            'Loaded model attention implementation differs from the frozen eager backend')
    attention_modules = [module for name, module in model.named_modules() if name.endswith('self_attn')]
    require(bool(attention_modules), 'No decoder self-attention modules found for backend verification')
    require(all(getattr(getattr(module, 'config', None), '_attn_implementation', None) == expected
                for module in attention_modules), 'Decoder attention implementation differs from the frozen eager backend')
    return expected


def configure_attention(model, config):
    """Select the recorded backend before the first forward through the model."""
    require(config.get('attention_implementation') == 'eager',
            'The frozen attention implementation must be eager')
    model.set_attn_implementation(config['attention_implementation'])
    return verify_attention(model, config)


def termination(token_ids, eos_token_ids, max_new_tokens):
    """EOS on the final permitted token is successful EOS termination, not a cap."""
    eos = bool(token_ids and token_ids[-1] in eos_token_ids)
    cap = len(token_ids) >= max_new_tokens and not eos
    return {'ended_with_eos': eos, 'cap_reached': cap,
            'finish_reason': 'eos' if eos else 'token_cap' if cap else 'other' if token_ids else 'empty_generation'}


def load_cached_tokenizer(bundle):
    """Resolve the exact cache commit offline and recheck original prompt tokens."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    config, expected = bundle['config'], bundle['manifest']['tokenizer']
    snapshot = Path(snapshot_download(config['model'], revision=config['model_revision'], local_files_only=True,
                    allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja']))
    require(snapshot.name == config['model_revision'], 'Cached model commit differs from pinned revision')
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    template = tokenizer.get_chat_template()
    provenance = {'model': config['model'], 'requested_revision': config['model_revision'],
                  'actual_snapshot_commit': snapshot.name, 'eos_token_id': tokenizer.eos_token_id,
                  'chat_template_sha256': sha256(template.encode()), 'class': type(tokenizer).__name__,
                  'local_files_only': True, 'trust_remote_code': False}
    for field in ('actual_snapshot_commit', 'eos_token_id', 'chat_template_sha256'):
        require(provenance[field] == expected[field], f'Cached tokenizer identity mismatch: {field}')
    if 'files' in expected:
        require(isinstance(expected['files'], dict) and bool(expected['files']), 'Invalid tokenizer file manifest')
        for name, digest in expected['files'].items():
            require(isinstance(name, str) and name == Path(name).name and is_digest(digest),
                    'Unsafe cached tokenizer file name or digest')
            require(sha256((snapshot / name).read_bytes()) == digest, f'Cached tokenizer file hash mismatch: {name}')
        provenance['files'] = expected['files']
    for row in bundle['rows']:
        messages = [{'role': 'system', 'content': row['system']}, {'role': 'user', 'content': row['request']}]
        actual = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        require(actual == prompt_inputs(row)[0], 'Exact prepared prompt token verification failed')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, snapshot, provenance


def verify_tools():
    import auditkit
    import safetune
    for name, package in (('SafeTune', safetune), ('AuditKIT', auditkit)):
        source = ROOT / 'vendor' / name
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        require(commit == TOOL_REVISIONS[name] and not dirty, f'{name} revision/cleanliness mismatch')
        require(Path(package.__file__).resolve().is_relative_to(source.resolve()), f'Unexpected {name} import source')
    return dict(TOOL_REVISIONS)


def nonempty_output(sample, output):
    """Execution diagnostic only: this is neither task accuracy nor safety."""
    return float(bool(output.strip()))


def ingest_auditkit(rows, records, output_path, provenance):
    """Use AuditKIT's supported precomputed-output API, with no second model."""
    import auditkit as ak
    from auditkit.model import PrecomputedModel
    require([r['id'] for r in rows] == [r['sample_id'] for r in records], 'Audit ingestion ID/order mismatch')
    samples = [ak.Sample(id=row['id'], input=row['request'], actual_output=record['response'],
                        metadata={key: row[key] for key in ('language', 'category', 'source_id', 'split')})
               for row, record in zip(rows, records)]
    result = ak.evaluate(samples, model=PrecomputedModel(),
                         scorers=[ak.scorer(direction=ak.Direction.MAXIMIZE)(nonempty_output)],
                         config=ak.RunConfig(seed=42, batch_size=1, concurrency=1, max_retries=0,
                                             track_performance=False, extra=provenance))
    result.metadata.update(diagnostic_only=True, semantic_review='pending',
                           scope='Nonempty-output execution diagnostic; no task accuracy or safety measurement.')
    result.save(str(output_path))
    require(not result.errors and result.failed_count == 0, 'AuditKIT diagnostic ingestion failed')
    require(len(result.predictions) == len(records), 'AuditKIT prediction count differs')
    expected = {r['sample_id']: r['response'] for r in records}
    require({p.sample_id: p.raw_output for p in result.predictions} == expected,
            'AuditKIT did not preserve the recorded raw responses')
    return {'status': 'passed_diagnostic_ingestion', 'fingerprint': result.fingerprint,
            'nonempty_output': result.headline['nonempty_output'], 'sha256': sha256(output_path.read_bytes()),
            'semantic_review': 'pending', 'diagnostic_only': True}


def generate_one(model, tokenizer, row, *, max_new_tokens, eos_token_ids, synchronize=lambda: None):
    import torch
    ids, mask = prompt_inputs(row)
    device = next(model.parameters()).device
    tensors = {'input_ids': torch.tensor([ids], dtype=torch.long, device=device),
               'attention_mask': torch.tensor([mask], dtype=torch.long, device=device)}
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**tensors, do_sample=False, num_beams=1,
                                max_new_tokens=max_new_tokens, use_cache=True,
                                eos_token_id=eos_token_ids, pad_token_id=tokenizer.pad_token_id,
                                return_dict_in_generate=False)
    synchronize()
    elapsed = time.perf_counter() - started
    require(output.shape[0] == 1 and output[0, :len(ids)].tolist() == ids,
            'Generation did not preserve the original prepared prompt')
    generated_ids = output[0, len(ids):].tolist()
    require(len(generated_ids) <= max_new_tokens, 'Generation exceeded its frozen token cap')
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    record = {key: row[key] for key in ('source_id', 'language', 'split', 'category', 'request', 'system')}
    if 'family_id' in row:
        record['family_id'] = row['family_id']
    if 'reference' in row:
        record['reference'] = row['reference']
    record.update(sample_id=row['id'], response=response, response_sha256=sha256(response.encode('utf-8')),
                  generated_token_ids=generated_ids, input_token_ids=ids,
                  input_token_ids_sha256=sha256(json_bytes(ids)), prepared_row_sha256=sha256(json_bytes(row)),
                  input_tokens=len(ids), output_tokens=len(generated_ids), nonempty=bool(response.strip()),
                  generation_seconds=round(elapsed, 6), **termination(generated_ids, eos_token_ids, max_new_tokens))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-run', type=Path)
    parser.add_argument('--split', choices=('development', 'alignment'), default='development')
    parser.add_argument('--limit', type=int, help='Fixed prefix of this split; always labeled feasibility-only')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing output directory')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'running', 'stage': 'validate_prepared',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'run_nonce': uuid.uuid4().hex,
              'split': args.split, 'feasibility_only': args.limit is not None, 'limit': args.limit,
              'generated_samples': 0, 'confirmation_generated_samples': 0, 'semantic_review': 'pending',
              'scope': 'Frozen public-data generation. Execution diagnostics do not establish task accuracy, safety, or improvement.',
              'source_hashes': {'experiments/multilingual_repair/evaluate.py': sha256(Path(__file__).read_bytes())}}

    def stage(name):
        report['stage'] = name
        write_json(args.output / 'report.json', report)
        print(name, flush=True)

    write_json(args.output / 'report.json', report)
    records = []
    try:
        bundle = load_prepared(args.prepared, args.split)
        config = bundle['config']
        report.update(prepared_manifest_sha256=bundle['manifest_sha256'], config_sha256=bundle['config_sha256'],
                      prepared_file_hashes=bundle['manifest']['files'], model=config['model'],
                      model_revision=config['model_revision'], generation={**config['generation'], 'seed': 42,
                      'batch_size': 1, 'do_sample': False, 'num_beams': 1})
        report['preparation_source_sha256'] = bundle['manifest'].get('preparation_script_sha256')
        report['public_source_hashes'] = {name: value['source_sha256']
            for name, value in bundle['manifest'].get('sources', {}).items()
            if isinstance(value, dict) and is_digest(value.get('source_sha256'))}
        (args.output / 'prepared-config.json').write_bytes(bundle['config_bytes'])
        training_run = load_training_run(args.training_run, bundle) if args.training_run else None
        report['variant'] = training_run['identity']['arm'] if training_run else 'base'
        if training_run:
            report['adapter_provenance'] = training_run['identity']
            report['feasibility_only'] = report['feasibility_only'] or training_run['identity']['feasibility_only']
        selected = bundle['rows'][:args.limit] if args.limit is not None else bundle['rows']
        report['selected_ids'] = [row['id'] for row in selected]
        report['selected_count'] = len(selected)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        stage('verify_runtime')
        import torch
        import safetune
        from safetune.runner.utils.model_utils import load_model
        report['tools'] = verify_tools()
        report['packages'] = {name: importlib.metadata.version(name)
                              for name in ('torch', 'transformers', 'tokenizers', 'safetune', 'auditkit', 'peft', 'huggingface-hub')}
        require(torch.backends.mps.is_available(), 'This experiment requires the frozen MPS runtime')
        torch.mps.set_per_process_memory_fraction(config['training']['mps_memory_fraction'])
        random.seed(42)
        torch.manual_seed(42)
        torch.mps.manual_seed(42)
        safetune.configure(device='mps', seed=42)
        stage('verify_tokenizer')
        tokenizer, snapshot, report['tokenizer'] = load_cached_tokenizer(bundle)
        stage('load_model')
        started = time.perf_counter()
        model = load_model(str(snapshot), dtype='bfloat16', device='mps')
        report['attention_implementation'] = configure_attention(model, config)
        if training_run:
            from peft import PeftModel
            fresh = load_training_run(args.training_run, bundle)
            require(fresh['identity'] == training_run['identity'], 'Training adapter changed during base loading')
            model = PeftModel.from_pretrained(model, str(fresh['adapter_path']), is_trainable=False,
                                              local_files_only=True)
            require(all(not p.requires_grad for p in model.parameters()), 'Inference adapter has trainable parameters')
            require(not any(getattr(module, 'merged', False) for module in model.modules()), 'Inference adapter was merged')
        model.eval()
        model.config.use_cache = True
        verify_attention(model, config)
        report['attention_implementation_verified'] = True
        report['attention_backend'] = {'requested': config['attention_implementation'],
            'effective': report['attention_implementation'],
            'attention_modules': sum(name.endswith('self_attn') for name, _ in model.named_modules())}
        torch.mps.synchronize()
        report.update(model_load_seconds=round(time.perf_counter() - started, 6),
                      model_device=str(next(model.parameters()).device), base_dtype=str(next(model.parameters()).dtype),
                      model_eval_mode=not model.training, adapter_merged=False)
        require(report['base_dtype'] == 'torch.bfloat16' and report['model_device'].startswith('mps'),
                'Loaded model dtype/device differs from frozen runtime')
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if type(eos_ids) is int else list(eos_ids or [])
        require(bool(eos_ids) and all(type(i) is int and i >= 0 for i in eos_ids)
                and tokenizer.eos_token_id in eos_ids, 'Model/tokenizer EOS configuration mismatch')
        report['eos_token_ids'] = eos_ids
        stage('generation')
        with (args.output / 'responses.jsonl').open('w', encoding='utf-8') as stream:
            for row in selected:
                record = generate_one(model, tokenizer, row, max_new_tokens=config['generation']['max_new_tokens'],
                                      eos_token_ids=eos_ids, synchronize=torch.mps.synchronize)
                record.update(prepared_manifest_sha256=bundle['manifest_sha256'], config_sha256=bundle['config_sha256'],
                              variant=report['variant'], feasibility_only=report['feasibility_only'],
                              attention_implementation=report['attention_implementation'])
                if training_run:
                    record['training_report_sha256'] = training_run['identity']['training_report_sha256']
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                records.append(record)
                report['generated_samples'] = len(records)
                write_json(args.output / 'report.json', report)
                print(f"{len(records)}/{len(selected)} {row['id']} tokens={record['output_tokens']} finish={record['finish_reason']}", flush=True)
        report['responses_sha256'] = sha256((args.output / 'responses.jsonl').read_bytes())
        report['diagnostics'] = {key: sum(r[key] for r in records)
                                 for key in ('nonempty', 'cap_reached', 'ended_with_eos', 'output_tokens')}
        stage('auditkit_ingestion')
        report['auditkit'] = ingest_auditkit(selected, records, args.output / 'auditkit-result.json', {
            'run_nonce': report['run_nonce'], 'responses_sha256': report['responses_sha256'],
            'prepared_manifest_sha256': bundle['manifest_sha256'], 'config_sha256': bundle['config_sha256'],
            'variant': report['variant'], 'feasibility_only': report['feasibility_only'],
            'attention_implementation': report['attention_implementation']})
        report['status'] = 'passed_execution'
        stage('complete')
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error_type=type(exc).__name__)
        # Avoid exporting runtime stack traces or absolute source/cache paths.
        print(f"{report['stage']} failed: {type(exc).__name__}", flush=True)
    finally:
        if (args.output / 'responses.jsonl').exists():
            report['responses_sha256'] = sha256((args.output / 'responses.jsonl').read_bytes())
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / 'report.json', report)
    return 0 if report['status'] == 'passed_execution' else 1


if __name__ == '__main__':
    raise SystemExit(main())
