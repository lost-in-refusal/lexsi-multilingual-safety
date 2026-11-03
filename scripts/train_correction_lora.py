#!/usr/bin/env python3
"""Train one fixed, adapter-only correction pilot from a verified token archive.

The complete policy stays in every input. Only the 32 training examples enter
optimizer updates. Calibration loss is diagnostic, not evidence of repair.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import time
import traceback

try:
    from .prepare_corrections import (_validate_config, load_cached_tokenizer,
                                     prepare_example, validate_correction_rows)
except ImportError:
    from prepare_corrections import (_validate_config, load_cached_tokenizer,
                                    prepare_example, validate_correction_rows)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {'train.jsonl', 'calibration.jsonl', 'examples.jsonl', 'policy.json',
             'config.json', 'runtime.json', 'chat-template.txt'}
INPUT_PATHS = {'data': 'data/corrections/examples.jsonl', 'policy': 'data/pilot/policy.json',
               'evaluation': 'data/pilot/scenarios.jsonl', 'config': 'configs/correction-lora.json',
               'runtime': 'configs/runtime.json'}
CORE_FIELDS = ('id', 'input_ids', 'attention_mask', 'labels', 'prompt_tokens',
               'sequence_tokens', 'supervised_tokens')
TARGETS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _read_lines(content):
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def _write_json(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def validate_training_config(config):
    """Reject settings unsupported by this fixed, one-epoch implementation."""
    expected = {
        'dtype': 'bfloat16', 'device': 'mps', 'seed': 42, 'train_steps': 32, 'epochs': 1,
        'batch_size': 1, 'gradient_accumulation_steps': 1, 'learning_rate': 0.0001,
        'weight_decay': 0.0, 'optimizer': 'AdamW', 'foreach': False, 'max_grad_norm': 1.0,
        'lora_r': 4, 'lora_alpha': 8, 'lora_dropout': 0.0, 'gradient_checkpointing': True,
        'use_reentrant': False, 'use_cache_during_training': False, 'max_sequence_tokens': 1024,
        'mps_memory_fraction': 0.8, 'loss': 'assistant_response_and_eos_only',
        'system_prompt': 'original_policy_only', 'save': 'adapter_only_no_merge_no_publication',
        'calibration': 'forward_only_token_weighted_loss_before_and_after_no_optimizer_or_checkpoint_selection',
    }
    _require(isinstance(config, dict), 'Training configuration must be an object')
    for key, value in expected.items():
        _require(type(config.get(key)) is type(value) and config[key] == value,
                 f'Unsupported fixed training setting: {key}')
    modules = config.get('target_modules')
    _require(isinstance(modules, list) and len(modules) == len(TARGETS)
             and all(isinstance(x, str) for x in modules) and set(modules) == TARGETS,
             'Expected exactly the seven frozen LoRA target modules')


def load_prepared(directory, root=ROOT):
    """Validate saved bytes and partition identities before token/model loading.

    Archived data is authoritative for training. Current canonical inputs must
    still match its recorded hashes; manifest-supplied external paths are not
    followed. Exact retokenization is a separate mandatory check before loading
    model weights.
    """
    directory, root = Path(directory), Path(root)
    manifest_bytes = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    _require(manifest.get('status') == 'passed_tokenization_preflight', 'Prepared archive has not passed preflight')
    _require(manifest.get('original_policy_only') is True and manifest.get('truncated_examples') == 0,
             'Expected original policy with zero truncation')
    _require(set(manifest.get('files', {})) == ARTIFACTS, 'Unexpected or unsafe archive file paths')
    blobs = {}
    for name in sorted(ARTIFACTS):
        blobs[name] = (directory / name).read_bytes()
        _require(_sha(blobs[name]) == manifest['files'][name], f'Archive file hash mismatch: {name}')
    _require(set(manifest.get('inputs', {})) == set(INPUT_PATHS), 'Prepared input identities differ')
    for name, relative in INPUT_PATHS.items():
        _require(_sha((root / relative).read_bytes()) == manifest['inputs'][name].get('sha256'),
                 f'Current canonical input hash differs: {name}')
    for name, filename in [('data', 'examples.jsonl'), ('policy', 'policy.json'),
                           ('config', 'config.json'), ('runtime', 'runtime.json')]:
        _require(_sha(blobs[filename]) == manifest['inputs'][name]['sha256'], f'Archived input hash differs: {name}')
    _require(_sha((root / 'scripts/prepare_corrections.py').read_bytes()) == manifest.get('script_sha256'),
             'Preflight implementation hash differs; prepare a new archive')
    config, runtime, policy = (json.loads(blobs[name]) for name in ('config.json', 'runtime.json', 'policy.json'))
    examples = _read_lines(blobs['examples.jsonl'])
    evaluation = _read_lines((root / INPUT_PATHS['evaluation']).read_bytes())
    validation = validate_correction_rows(examples, policy, evaluation)
    _require(validation == manifest.get('validation'), 'Saved validation differs from actual authored examples')
    _validate_config(config, runtime, validation, _sha(blobs['policy.json']))
    validate_training_config(config)
    _require(manifest.get('max_sequence_tokens') == config['max_sequence_tokens'], 'Sequence limit differs')
    _require(_sha(blobs['chat-template.txt']) == manifest['tokenizer'].get('chat_template_sha256'),
             'Saved chat template hash mismatch')
    _require(manifest['tokenizer'].get('actual_snapshot_commit') == config['model_revision'], 'Tokenizer revision mismatch')
    examples_by_id = {row['id']: row for row in examples}
    splits = {}
    seen = set()
    for split in ('train', 'calibration'):
        records = _read_lines(blobs[split + '.jsonl'])
        expected_ids = [row['id'] for row in examples if row['split'] == split]
        _require([row.get('id') for row in records] == expected_ids, f'{split} IDs/order differ from authored partition')
        for row in records:
            sample_id = row['id']
            _require(sample_id not in seen, 'Duplicate token record across partitions')
            seen.add(sample_id)
            source = examples_by_id[sample_id]
            _require(set(row) == set(CORE_FIELDS) | {'family_id', 'split', 'language', 'category'},
                     f'{sample_id}: unexpected token record fields')
            for key in ('family_id', 'split', 'language', 'category'):
                _require(row[key] == source[key], f'{sample_id}: token metadata mismatch: {key}')
            ids, labels, mask = row['input_ids'], row['labels'], row['attention_mask']
            n, prefix, supervised = row['sequence_tokens'], row['prompt_tokens'], row['supervised_tokens']
            _require(all(type(v) is int for v in (n, prefix, supervised)) and 0 < prefix < n <= config['max_sequence_tokens'],
                     f'{sample_id}: invalid token counts')
            _require(isinstance(ids, list) and len(ids) == n and all(type(v) is int and v >= 0 for v in ids),
                     f'{sample_id}: invalid input IDs')
            _require(type(manifest['tokenizer'].get('eos_token_id')) is int and ids[-1] == manifest['tokenizer']['eos_token_id'],
                     f'{sample_id}: missing terminal EOS')
            _require(isinstance(labels, list) and len(labels) == n and all(type(v) is int for v in labels)
                     and labels == [-100] * prefix + ids[prefix:], f'{sample_id}: incorrect assistant-only labels')
            _require(isinstance(mask, list) and all(type(v) is int for v in mask) and mask == [1] * n,
                     f'{sample_id}: invalid attention mask')
            _require(supervised == n - prefix == sum(v != -100 for v in labels[1:]) and supervised >= 2,
                     f'{sample_id}: incorrect supervised-token count')
        splits[split] = records
    _require(seen == set(examples_by_id), 'Prepared partition coverage differs')
    return {'directory': directory, 'manifest': manifest, 'manifest_sha256': _sha(manifest_bytes),
            'config': config, 'runtime': runtime, 'policy': policy, 'examples': examples,
            'blobs': blobs, 'validation': validation, **splits}


def verify_retokenized(bundle):
    """Compare every saved tensor input with the fresh pinned cached tokenizer."""
    tokenizer, template, provenance = load_cached_tokenizer(bundle['config'])
    expected = bundle['manifest']['tokenizer']
    for key in ('model', 'requested_revision', 'actual_snapshot_commit', 'class', 'eos_token_id',
                'chat_template_sha256', 'files', 'packages', 'local_files_only', 'trust_remote_code'):
        _require(provenance.get(key) == expected.get(key), f'Cached tokenizer identity mismatch: {key}')
    _require(template.encode('utf-8') == bundle['blobs']['chat-template.txt'], 'Fresh tokenizer template differs')
    saved = {row['id']: row for row in bundle['train'] + bundle['calibration']}
    for row in bundle['examples']:
        actual = prepare_example(row, bundle['policy'], tokenizer, bundle['config']['max_sequence_tokens'])
        _require(actual == {key: saved[row['id']][key] for key in CORE_FIELDS},
                 f"Exact retokenization differs for {row['id']}")
    return tokenizer, provenance


def training_order(records, *, seed, steps):
    _require(type(seed) is int and type(steps) is int and steps > 0, 'Seed and steps must be integers')
    _require(isinstance(records, list) and len(records) == steps, 'Steps must cover one complete training epoch')
    _require(all(r.get('split') == 'train' for r in records), 'Only train records can enter optimizer order')
    ids = [r.get('id') for r in records]
    _require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids), 'Training IDs must be unique')
    random.Random(seed).shuffle(ids)
    return ids


def aggregate_calibration(rows):
    _require(isinstance(rows, list) and bool(rows), 'Calibration needs measured rows')
    ids = set()
    total, weighted = 0, 0.0
    for row in rows:
        identity, loss, count = row.get('id'), row.get('loss'), row.get('supervised_tokens')
        _require(isinstance(identity, str) and identity and identity not in ids, 'Duplicate or invalid calibration ID')
        _require(type(loss) in (int, float) and math.isfinite(loss) and loss >= 0, 'Calibration loss must be finite and nonnegative')
        _require(type(count) is int and count > 0, 'Calibration supervised-token count must be positive')
        ids.add(identity); total += count; weighted += loss * count
    _require(math.isfinite(weighted), 'Nonfinite accumulated calibration loss')
    return {'rows': len(rows), 'supervised_tokens': total, 'token_weighted_loss': weighted / total,
            'measurements': rows, 'scope': 'Forward-only optimization diagnostic; no checkpoint selection or safety claim.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Validate archive and tokenizer without model weights or training')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'Refusing to overwrite {args.output}')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'stage': 'validate_prepared', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'One fixed plain LoRA correction pilot; process success and loss do not establish behavior repair.',
              'optimizer_steps': 0, 'model_weights_loaded': False, 'steps': [], 'memory_snapshots': [],
              'test_generated_samples': 0, 'test_scored_samples': 0, 'human_review_status': 'pending',
              'memory_preflight': {'status': 'not_run'}, 'adapter_directory': 'adapter'}

    def write_report():
        _write_json(args.output / 'report.json', report)

    def stage(name):
        report['stage'] = name
        write_report()
        print(name, flush=True)

    write_report()
    try:
        bundle = load_prepared(args.prepared)
        config = bundle['config']
        report.update(model=config['model'], model_revision=config['model_revision'], training_config=config,
                      training_config_sha256=_sha(bundle['blobs']['config.json']),
                      prepared_directory=str(args.prepared.resolve()), prepared_manifest_sha256=bundle['manifest_sha256'],
                      prepared_file_hashes=bundle['manifest']['files'], input_hashes=bundle['manifest']['inputs'],
                      data_partition_summary=bundle['validation'], source_hashes={
                          'scripts/train_correction_lora.py': _sha(Path(__file__).read_bytes()),
                          'scripts/prepare_corrections.py': _sha((ROOT / 'scripts/prepare_corrections.py').read_bytes()),
                          'configs/tools.lock.json': _sha((ROOT / 'configs/tools.lock.json').read_bytes())})
        (args.output / 'training-config.json').write_bytes(bundle['blobs']['config.json'])
        (args.output / 'prepared-manifest.json').write_bytes((args.prepared / 'manifest.json').read_bytes())
        stage('verify_exact_tokenization')
        tokenizer, provenance = verify_retokenized(bundle)
        report['tokenizer'] = provenance
        report['exact_retokenization_verified'] = len(bundle['examples'])
        order = training_order(bundle['train'], seed=config['seed'], steps=config['train_steps'])
        report['training_order'] = order
        report['calibration_example_ids'] = [r['id'] for r in bundle['calibration']]
        if args.prepare_only:
            report.update(status='prepared_not_trained', stage='complete')
            return 0

        import torch
        import safetune
        from peft import get_peft_model_state_dict
        from safetune.runner.utils.model_utils import load_model, lora_wrap

        stage('verify_training_environment')
        source = ROOT / 'vendor/SafeTune'
        pins = json.loads((ROOT / 'configs/tools.lock.json').read_text())
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        _require(commit == pins['SafeTune']['revision'] and not dirty, 'SafeTune source revision/cleanliness mismatch')
        _require(Path(safetune.__file__).resolve().is_relative_to(source.resolve()), 'Unexpected SafeTune import source')
        _require(torch.backends.mps.is_available(), 'Configured MPS backend is unavailable')
        report['tools'] = {'SafeTune': commit}
        report['packages'] = {name: importlib.metadata.version(name) for name in
                              ('torch', 'transformers', 'peft', 'safetune', 'huggingface-hub', 'tokenizers')}
        report['mps_recommended_max_gib'] = torch.mps.recommended_max_memory() / 2**30
        report['mps_memory_fraction'] = config['mps_memory_fraction']
        report['initial_system_swap'] = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
        torch.mps.set_per_process_memory_fraction(config['mps_memory_fraction'])

        def reset_seed():
            safetune.configure(device='mps', seed=config['seed'])
            random.seed(config['seed'])
            torch.manual_seed(config['seed'])
            torch.mps.manual_seed(config['seed'])

        def memory(name):
            torch.mps.synchronize()
            report['memory_snapshots'].append({'stage': name,
                'mps_tensor_gib': round(torch.mps.current_allocated_memory() / 2**30, 3),
                'mps_driver_gib': round(torch.mps.driver_allocated_memory() / 2**30, 3),
                'process_max_rss_gib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 3)})
            write_report()

        reset_seed()
        stage('load_model')
        started = time.perf_counter()
        model = load_model(provenance['snapshot_path'], dtype=config['dtype'], device=config['device'])
        torch.mps.synchronize()
        report['model_weights_loaded'] = True
        report['model_load_seconds'] = round(time.perf_counter() - started, 3)
        report['model_device'] = str(next(model.parameters()).device)
        report['base_dtype'] = str(next(model.parameters()).dtype)
        _require(report['model_device'].split(':')[0] == 'mps' and report['base_dtype'] == 'torch.bfloat16', 'Loaded device/dtype differs')
        memory('base_loaded')
        stage('create_lora_adapter')
        model = lora_wrap(model, r=config['lora_r'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                          target_modules=config['target_modules'])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': config['use_reentrant']})
        model.enable_input_require_grads()
        _require(model.is_gradient_checkpointing, 'Gradient checkpointing was not enabled')
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        _require(bool(trainable) and all('lora_' in name for name in trainable), 'Unexpected trainable base parameters')
        report['trainable_parameters'] = sum(p.numel() for p in trainable.values())
        report['trainable_dtypes'] = sorted({str(p.dtype) for p in trainable.values()})
        _require(report['trainable_dtypes'] == ['torch.float32'], 'Expected FP32 trainable adapters')
        report['target_modules'] = sorted(model.peft_config['default'].target_modules)
        _require(set(report['target_modules']) == TARGETS, 'Actual adapter target modules differ')
        report['all_base_parameters_frozen'] = all(not p.requires_grad for name, p in model.named_parameters() if 'lora_' not in name)
        _require(report['all_base_parameters_frozen'], 'Unfrozen base weights')
        initial_adapter = {name: p.detach().cpu().clone() for name, p in trainable.items()}
        frozen = [(name, p, p.detach().flatten()[:1024].cpu().clone()) for name, p in model.named_parameters() if not p.requires_grad]
        report['frozen_parameter_samples_checked'] = len(frozen)
        report['frozen_parameter_check_scope'] = 'First 1024 elements of each frozen tensor, plus requires_grad/gradient checks; not a full-weight hash.'
        memory('adapter_ready')

        def batch_for(row):
            return {key: torch.tensor([row[key]], dtype=torch.long, device='mps')
                    for key in ('input_ids', 'attention_mask', 'labels')}

        def gradient_norm():
            grads = [p.grad for p in trainable.values() if p.grad is not None]
            _require(bool(grads) and all(torch.isfinite(g).all().item() for g in grads), 'Missing or nonfinite adapter gradients')
            _require(all(p.grad is None for _, p, _ in frozen), 'Frozen base received gradients')
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), config['max_grad_norm'], error_if_nonfinite=True).item()
            _require(math.isfinite(norm) and norm > 0, 'Missing/nonfinite/zero training gradient norm')
            return norm

        stage('longest_example_memory_preflight')
        longest = max(bundle['train'], key=lambda r: r['sequence_tokens'])
        model.train(); model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        batch = batch_for(longest)
        loss = model(**batch, use_cache=False).loss
        _require(torch.isfinite(loss).item(), 'Nonfinite memory-preflight loss')
        loss_value = loss.item(); memory('memory_preflight_forward')
        loss.backward(); norm = gradient_norm(); memory('memory_preflight_backward')
        model.zero_grad(set_to_none=True)
        _require(all(torch.equal(initial_adapter[name], p.detach().cpu()) for name, p in trainable.items()),
                 'Memory preflight changed adapter state without optimizer updates')
        report['memory_preflight'] = {'status': 'passed', 'sample_id': longest['id'], 'sequence_tokens': longest['sequence_tokens'],
            'supervised_tokens': longest['supervised_tokens'], 'loss': loss_value, 'gradient_norm_before_clip': norm,
            'optimizer_steps': 0, 'adapter_state_unchanged': True, 'seconds': round(time.perf_counter() - started, 3)}
        del loss, batch
        gc.collect(); torch.mps.empty_cache()
        reset_seed()
        memory('memory_preflight_cleared')

        def calibrate(label, *, disable_adapter=False):
            stage(label); model.eval(); model.zero_grad(set_to_none=True)
            measurements = []
            context = model.disable_adapter() if disable_adapter else nullcontext()
            with context, torch.no_grad():
                for index, row in enumerate(bundle['calibration'], 1):
                    batch = batch_for(row)
                    loss = model(**batch, use_cache=False).loss
                    _require(torch.isfinite(loss).item(), 'Nonfinite calibration loss')
                    measurements.append({'id': row['id'], 'loss': loss.item(), 'supervised_tokens': row['supervised_tokens']})
                    del batch, loss
                    print(f'{label} {index}/{len(bundle["calibration"])}', flush=True)
            result = aggregate_calibration(measurements)
            result['adapter_enabled'] = not disable_adapter
            memory(label)
            return result

        report['calibration_before'] = calibrate('calibration_before', disable_adapter=True)
        reset_seed()
        _require(order == training_order(bundle['train'], seed=config['seed'], steps=config['train_steps']), 'Training order drift')
        by_id = {row['id']: row for row in bundle['train']}
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=config['learning_rate'], weight_decay=config['weight_decay'], foreach=False)
        started_training = time.perf_counter()
        with (args.output / 'steps.jsonl').open('x') as steps_file:
            for step_number, sample_id in enumerate(order, 1):
                stage(f'training_step_{step_number}')
                row = by_id[sample_id]
                _require(row['split'] == 'train', 'Calibration row entered optimizer')
                model.train(); optimizer.zero_grad(set_to_none=True)
                started = time.perf_counter()
                batch = batch_for(row)
                loss = model(**batch, use_cache=False).loss
                _require(torch.isfinite(loss).item(), 'Nonfinite training loss')
                loss_value = loss.item(); memory(f'step_{step_number}_forward')
                loss.backward(); norm = gradient_norm(); memory(f'step_{step_number}_backward')
                optimizer.step()
                report['optimizer_steps'] += 1
                _require(all(torch.isfinite(p).all().item() for p in trainable.values()), 'Nonfinite adapter weights')
                step = {'step': step_number, 'sample_id': sample_id, 'split': row['split'],
                        'loss': loss_value, 'supervised_tokens': row['supervised_tokens'],
                        'gradient_norm_before_clip': norm, 'finite_gradients': True,
                        'seconds': round(time.perf_counter() - started, 3)}
                report['steps'].append(step)
                steps_file.write(json.dumps(step, allow_nan=False) + '\n'); steps_file.flush(); os.fsync(steps_file.fileno())
                memory(f'step_{step_number}_optimizer')
                del loss, batch
        report['training_seconds'] = round(time.perf_counter() - started_training, 3)
        _require(report['optimizer_steps'] == config['train_steps'] == 32 and [x['sample_id'] for x in report['steps']] == order,
                 'Optimizer did not execute exactly the complete frozen epoch')
        changes = [(p.detach().cpu() - initial_adapter[name]).abs().max().item() for name, p in trainable.items()]
        report['changed_adapter_tensors'] = sum(value > 0 for value in changes)
        report['maximum_adapter_absolute_change'] = max(changes)
        _require(report['changed_adapter_tensors'] > 0, 'Adapter never updated')
        _require(all(not p.requires_grad and p.grad is None for _, p, _ in frozen), 'Frozen base state changed during training')
        report['frozen_parameter_samples_unchanged'] = all(torch.equal(saved, p.detach().flatten()[:1024].cpu()) for _, p, saved in frozen)
        _require(report['frozen_parameter_samples_unchanged'], 'A sampled frozen parameter changed')
        optimizer.zero_grad(set_to_none=True)
        del optimizer, initial_adapter
        gc.collect(); torch.mps.empty_cache()
        model.gradient_checkpointing_disable()
        report['calibration_after'] = calibrate('calibration_after')

        stage('save_and_reload_adapter')
        adapter = args.output / report['adapter_directory']
        model.peft_config['default'].base_model_name_or_path = config['model']
        model.peft_config['default'].revision = config['model_revision']
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        original_state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model, adapter_name='default').items()}
        model.load_adapter(adapter, adapter_name='reloaded', is_trainable=False)
        reloaded_state = get_peft_model_state_dict(model, adapter_name='reloaded')
        report['saved_adapter_reload_exact'] = original_state.keys() == reloaded_state.keys() and all(
            torch.equal(value, reloaded_state[key].detach().cpu()) for key, value in original_state.items())
        _require(report['saved_adapter_reload_exact'], 'Reloaded adapter tensors differ')
        model.set_adapter('reloaded'); model.eval()
        report['calibration_reloaded'] = calibrate('calibration_reloaded')
        _require([r['id'] for r in report['calibration_after']['measurements']] == [r['id'] for r in report['calibration_reloaded']['measurements']],
                 'Reload calibration row identities differ')
        tolerance = {'relative': 1e-5, 'absolute': 1e-6}
        report['reload_loss_tolerance'] = tolerance
        report['saved_adapter_reload_loss_matches'] = all(math.isclose(a['loss'], b['loss'], rel_tol=tolerance['relative'], abs_tol=tolerance['absolute'])
            for a, b in zip(report['calibration_after']['measurements'], report['calibration_reloaded']['measurements']))
        _require(report['saved_adapter_reload_loss_matches'], 'Reloaded adapter calibration losses differ beyond tolerance')
        report['adapter_files'] = {str(p.relative_to(adapter)): _sha(p.read_bytes()) for p in sorted(adapter.rglob('*')) if p.is_file()}
        _require({'adapter_config.json', 'adapter_model.safetensors'} <= set(report['adapter_files']), 'Adapter-only output files missing')
        report['adapter_weights_bytes'] = (adapter / 'adapter_model.safetensors').stat().st_size
        report['maximum_observed_stage_mps_driver_gib'] = max(x['mps_driver_gib'] for x in report['memory_snapshots'])
        report['memory_measurement_scope'] = 'Synchronized stage-boundary MPS samples, not a continuous/guaranteed peak or total system memory. Process RSS is an OS high-water reading.'
        report['final_system_swap'] = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
        report.update(status='passed_training', stage='complete')
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error_type=type(exc).__name__, error=str(exc))
        (args.output / 'error.log').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_report()
        print(f"{report['status'].upper()}: {args.output / 'report.json'}", flush=True)
    return 0 if report['status'] == 'passed_training' else 1


if __name__ == '__main__':
    raise SystemExit(main())
