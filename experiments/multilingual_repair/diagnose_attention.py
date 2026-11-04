"""Compare SDPA/eager logits with and without autograd; never update weights.

Run with ``python -m experiments.multilingual_repair.diagnose_attention
--prepared PREPARED_DIRECTORY --output NEW_OUTPUT_DIRECTORY``.
Only alignment rows are read. KL is the raw pinned helper result, including its
roundoff residual at equal logits. The historical ``max_driver_gib`` observation
was a final allocated-memory sample, not a measured peak; this command names that
measurement ``final_mps_driver_allocated_gib``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path

from experiments.multilingual_repair import evaluate, train

MODES = ('disabled_nograd', 'active_nograd', 'active_grad', 'disabled_nograd_repeat')


def load_alignment_bundle(directory):
    """Verify config/alignment bytes; leave every other partition unread."""
    directory = Path(directory)
    manifest_bytes = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    train.require(manifest.get('status') == 'passed_tokenization_preflight'
                  and manifest.get('truncated_examples') == 0, 'Prepared preflight did not pass')
    files = manifest.get('files', {})
    train.require(set(files) == train.REQUIRED_FILES
                  and all(evaluate.is_digest(value) for value in files.values()),
                  'Unexpected prepared file names or digests')
    checked = {}
    for name in ('config.json', 'alignment.jsonl'):
        path = directory / name
        train.require(path.is_file() and not path.is_symlink(), 'Missing or symlinked diagnostic input')
        checked[name] = path.read_bytes()
        train.require(train.sha_bytes(checked[name]) == files[name], f'Prepared hash mismatch: {name}')
    config = json.loads(checked['config.json'])
    train.validate_config(config)
    tokenizer = manifest.get('tokenizer', {})
    train.require(tokenizer.get('actual_snapshot_commit') == train.REVISION
                  and type(tokenizer.get('eos_token_id')) is int
                  and evaluate.is_digest(tokenizer.get('chat_template_sha256')),
                  'Prepared tokenizer identity is missing or mismatched')
    rows = [json.loads(line) for line in checked['alignment.jsonl'].splitlines() if line.strip()]
    seen = set()
    for row in rows:
        train.require(isinstance(row, dict) and isinstance(row.get('id'), str) and row['id']
                      and row['id'] not in seen, 'Missing or duplicate alignment identity')
        seen.add(row['id'])
        train.require(row.get('split') == 'alignment' and row.get('category') == 'disallowed'
                      and all(isinstance(row.get(key), str) and row[key]
                              for key in ('source_id', 'request', 'system')), 'Invalid alignment row')
        train.validate_token_row(row, max_tokens=config['training']['max_sequence_tokens'],
                                 eos_token_id=tokenizer['eos_token_id'], alignment=True)
    train.require(Counter(row.get('language') for row in rows) == {'en': 24, 'ko': 24, 'zh': 24},
                  'Expected the complete 72-prompt alignment partition')
    return {'manifest': manifest, 'manifest_sha256': train.sha_bytes(manifest_bytes),
            'config': config, 'config_sha256': files['config.json'], 'rows': rows,
            'checked_prepared_files': {name: files[name] for name in checked}}


def compare_logits(reference, candidate, kl_helper):
    """Portable CPU comparison: raw KL(reference || candidate), batchmean."""
    import torch
    train.require(reference.shape == candidate.shape and reference.ndim == 3
                  and reference.shape[0] == 1 and reference.numel() > 0, 'Invalid comparison shape')
    reference, candidate = reference.detach().cpu().float(), candidate.detach().cpu().float()
    train.require(torch.isfinite(reference).all().item() and torch.isfinite(candidate).all().item(),
                  'Nonfinite diagnostic logits')
    difference = (reference - candidate).abs()
    metrics = {'equal': torch.equal(reference, candidate), 'max_abs': difference.max().item(),
               'mean_abs': difference.mean().item(),
               'kl_batchmean': kl_helper(candidate, reference, temperature=1., reduction='batchmean').item()}
    train.require(all(math.isfinite(value) for key, value in metrics.items() if key != 'equal'),
                  'Nonfinite diagnostic metric')
    return metrics


def compare_modes(outputs, kl_helper):
    import torch
    train.require(set(outputs) == set(MODES), 'All four execution modes are required')
    reference = outputs['disabled_nograd']
    return {'comparisons': {mode: compare_logits(reference, outputs[mode], kl_helper) for mode in MODES},
            'active_grad_vs_nograd_equal': torch.equal(outputs['active_grad'], outputs['active_nograd'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    args.output.mkdir(parents=True)
    report = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'optimizer_steps': 0, 'split': 'alignment', 'backends': {},
              'scope': 'Numerical backend diagnostic; no training or semantic evaluation.',
              'kl_scope': 'Raw pinned KL(reference||candidate), temperature1, batchmean; summed positions at batch1.',
              'memory_measurement_scope': 'Final synchronized driver allocation after cleanup; not a peak.',
              'source_hashes': {f'experiments/multilingual_repair/{name}': train.sha_bytes(
                  Path(__file__).with_name(name).read_bytes())
                  for name in ('diagnose_attention.py', 'train.py', 'evaluate.py')}}
    train.write_json(args.output / 'report.json', report)
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        bundle = load_alignment_bundle(args.prepared)
        config = bundle['config']
        report.update(model=config['model'], model_revision=config['model_revision'],
                      config_sha256=bundle['config_sha256'], prepared_manifest_sha256=bundle['manifest_sha256'],
                      checked_prepared_files=bundle['checked_prepared_files'],
                      unread_partitions=['train', 'calibration', 'development', 'confirmation'],
                      configured_training_attention=config['attention_implementation'])
        report['environment'] = train.verify_environment()
        tokenizer, snapshot, report['tokenizer'] = evaluate.load_cached_tokenizer(bundle)
        del tokenizer
        import torch
        import safetune
        from safetune.runner.utils.model_utils import load_model, lora_wrap
        from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
        train.require(torch.backends.mps.is_available(), 'MPS is required for this device diagnostic')
        torch.mps.set_per_process_memory_fraction(config['training']['mps_memory_fraction'])
        safetune.configure(device='mps', seed=config['seed'])
        torch.manual_seed(config['seed']); torch.mps.manual_seed(config['seed'])
        model = load_model(str(snapshot), dtype='bfloat16', device='mps')
        settings = config['training']
        model = lora_wrap(model, r=settings['lora_r'], lora_alpha=settings['lora_alpha'],
                          lora_dropout=settings['lora_dropout'], target_modules=settings['target_modules'])
        model.enable_input_require_grads()
        model.eval()
        adapters = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
        train.require(adapters and all('lora_' in name for name in adapters), 'Unexpected trainable parameters')
        initial_b = [p for name, p in model.named_parameters() if '.lora_B.' in name]
        report['initial_lora_B_zero'] = bool(initial_b) and all(torch.count_nonzero(p).item() == 0 for p in initial_b)
        train.require(report['initial_lora_B_zero'], 'Diagnostic requires zero-initialized LoRA B tensors')
        row = max(bundle['rows'], key=lambda item: item['sequence_tokens'])
        batch = {key: torch.tensor([row[key]], dtype=torch.long, device='mps')
                 for key in ('input_ids', 'attention_mask')}
        report.update(sample_id=row['id'], sequence_tokens=row['sequence_tokens'],
                      selection='Longest frozen alignment sequence; first prepared row wins ties.',
                      loaded_attention_implementation=model.config._attn_implementation,
                      dropouts=sorted({module.p for module in model.modules() if isinstance(module, torch.nn.Dropout)}))
        for backend in ('sdpa', 'eager'):
            model.get_base_model().set_attn_implementation(backend)
            attention = [module for name, module in model.named_modules() if name.endswith('self_attn')]
            train.require(model.config._attn_implementation == backend and attention
                          and all(module.config._attn_implementation == backend for module in attention),
                          'Requested diagnostic backend was not applied')
            outputs = {}
            for mode in MODES:
                if mode.startswith('disabled'):
                    with train.disabled_adapter_reference(model), torch.no_grad():
                        logits = model(**batch, use_cache=False).logits
                elif mode == 'active_nograd':
                    with torch.no_grad():
                        logits = model(**batch, use_cache=False).logits
                else:
                    with torch.enable_grad():
                        logits = model(**batch, use_cache=False).logits
                    train.require(logits.requires_grad, 'Student diagnostic must use the autograd-enabled path')
                outputs[mode] = logits.detach().cpu().float()
                del logits
                gc.collect(); torch.mps.empty_cache()
            report['backends'][backend] = compare_modes(outputs, safegrad_kl_alignment_loss)
            del outputs
            train.write_json(args.output / 'report.json', report)
        report['adapter_state_unchanged'] = all(torch.equal(adapters[name], p.detach().cpu())
                                              for name, p in model.named_parameters() if name in adapters)
        train.require(report['adapter_state_unchanged'] and all(p.grad is None for p in model.parameters()),
                      'Diagnostic unexpectedly mutated adapter state or populated gradients')
        gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()
        report['final_mps_driver_allocated_gib'] = torch.mps.driver_allocated_memory() / 2**30
        report['status'] = 'passed_diagnostic'
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__)
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        train.write_json(args.output / 'report.json', report)
    return 0 if report['status'] == 'passed_diagnostic' else 1


if __name__ == '__main__':
    raise SystemExit(main())
