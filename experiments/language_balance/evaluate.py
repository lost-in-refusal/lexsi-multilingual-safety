"""Greedy development generation for the frozen language exposure comparison.

Numerical, tokenization, attention, generation and ingestion helpers are imported
unchanged from v1. Confirmation is unavailable. References stay outside inputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import uuid
from experiments.multilingual_repair import evaluate as frozen
from experiments.multilingual_repair.evaluate import (
    require, sha256, json_bytes, write_json, is_digest, load_prepared, prompt_inputs,
    load_cached_tokenizer, generate_one, termination, verify_tools,
    configure_attention, verify_attention, ingest_auditkit)
from .study import load_study


def validate_development(bundle):
    rows = bundle['rows']
    require(bundle['config']['generation']['max_new_tokens'] == 512,
            'Frozen development generation requires the 512 token cap')
    require(len(rows) == 72 and len({r['id'] for r in rows}) == 72,
            'Expected exactly 72 unique development cases')
    require(Counter((r['language'], r['category']) for r in rows) ==
            Counter({(l, c): 12 for l in ('en', 'ko', 'zh') for c in ('task', 'disallowed')}),
            'Expected 12 task and 12 safety cases per language')
    families = {}
    for row in rows:
        if row['category'] == 'disallowed':
            families.setdefault(row.get('family_id'), []).append(row['language'])
    require(len(families) == 12 and all(sorted(ls) == ['en','ko','zh'] for ls in families.values()),
            'Expected 12 aligned safety source families')


def validate_checkpoint(snapshot, training_run):
    from .train import checkpoint_hashes
    require(checkpoint_hashes(snapshot) == training_run['identity']['base_checkpoint_files'],
            'Inference base checkpoint bytes differ from frozen training checkpoint')


def load_training_run(directory, bundle, study):
    from .train import load_training_run as validate_training
    adapter, report = validate_training(directory, study, bundle=bundle)
    require(report.get('status') == 'passed_training' and report.get('run_kind') == 'fixed_schedule'
            and report.get('training_method') == 'safegrad' and report.get('arm') in ('control', 'balanced'),
            'Evaluation requires a complete fixed_schedule control or balanced SafeGrad run')
    require(report.get('expected_steps') == report.get('optimizer_steps') == 1000,
            'Incomplete fixed training schedule')
    require(report.get('study_manifest_sha256') == study['manifest_sha256'], 'Training study identity differs')
    require(isinstance(report.get('initialization'), dict)
            and is_digest(report['initialization'].get('manifest_sha256')), 'Missing shared initialization identity')
    identity = {**report, 'training_report_sha256': sha256((Path(directory)/'report.json').read_bytes()),
                'feasibility_only': False}
    # Directory locations are runtime details; the bound bytes are the identity.
    identity['initialization'] = {k:v for k,v in report['initialization'].items() if k != 'directory'}
    return {'adapter_path': adapter, 'identity': identity}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Fixed prefix of this split; always labeled feasibility-only')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing output directory')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 2, 'experiment':'language_balance', 'status': 'running', 'stage': 'validate_prepared',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'run_nonce': uuid.uuid4().hex,
              'split': 'development', 'feasibility_only': args.limit is not None, 'limit': args.limit,
              'generated_samples': 0, 'confirmation_generated_samples': 0, 'semantic_review': 'pending',
              'scope': 'Frozen public-data generation. Execution diagnostics do not establish task accuracy, safety, or improvement.',
              'source_hashes': {'experiments/language_balance/evaluate.py': sha256(Path(__file__).read_bytes()),
                  'experiments/multilingual_repair/evaluate.py': sha256(Path(frozen.__file__).read_bytes())}}

    def stage(name):
        report['stage'] = name
        write_json(args.output / 'report.json', report)
        print(name, flush=True)

    write_json(args.output / 'report.json', report)
    records = []
    try:
        bundle = load_prepared(args.prepared, 'development')
        validate_development(bundle)
        study = load_study(args.study, args.prepared)
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
        training_run = load_training_run(args.training_run, bundle, study) if args.training_run else None
        report['study_manifest_sha256'] = study['manifest_sha256']
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
        validate_checkpoint(snapshot, training_run)
        stage('load_model')
        started = time.perf_counter()
        model = load_model(str(snapshot), dtype='bfloat16', device='mps')
        report['attention_implementation'] = configure_attention(model, config)
        if training_run:
            from peft import PeftModel
            fresh = load_training_run(args.training_run, bundle, study)
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
