"""Pinned, local Qwen3Guard screening of complete Tiny Aya candidate families.

Only completed strict screening yields optimizer/evaluation rows. The journal
keeps raw moderator text and explicit pending checks; no human labels are inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import time

from .prepare import CONFIG, LANGUAGES, verify_bundle

CATEGORIES = frozenset(('Violent', 'Non-violent Illegal Acts', 'Sexual Content or Sexual Acts',
                        'PII', 'Suicide & Self-Harm', 'Unethical Acts',
                        'Politically Sensitive Topics', 'Copyright Violation', 'Jailbreak'))
SPLITS = ('train.jsonl', 'eval_task.jsonl', 'training_qc.jsonl')
SETTINGS = dict(schema_version=1, strict=True, passing_families=700, eval_families=100,
                train_families=600, prompt_cap=1024, suffix_cap=512, sequence_cap=2048,
                guard_input_cap=4096, max_new_tokens=128, do_sample=False,
                device='mps', dtype='bfloat16', attention='eager', memory_fraction=0.8,
                model=CONFIG['model'], model_revision=CONFIG['model_revision'],
                guard=CONFIG['guard'], guard_revision=CONFIG['guard_revision'])


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return _sha(_json(value).encode())


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    """Atomic final artifacts; journal has its own append/fsync boundary."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(_json(value) + '\n'); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def parse_completion(raw, mode, *, eos=True, generated_tokens=1, input_tokens=1, stop_reason='eos'):
    """Parse only complete official line-form labels; ambiguity never passes.

    Strict requires Safe; loose permits Controversial. Both require non-refusal
    for response moderation. Neither interpretation includes unscorable output.
    """
    result = dict(status='unscorable', safety=None, categories=None, refusal=None,
                  strict_pass=False, loose_pass=False)
    if mode not in ('prompt', 'response'):
        raise ValueError('mode must be prompt or response')
    if (not isinstance(raw, str) or eos is not True or stop_reason != 'eos'
            or type(generated_tokens) is not int or not 0 < generated_tokens < 128
            or type(input_tokens) is not int or not 0 < input_tokens <= 4096):
        return result
    pattern = r'Safety: (Safe|Unsafe|Controversial)\nCategories: ([^\n]+)'
    if mode == 'response': pattern += r'\nRefusal: (Yes|No)'
    match = re.fullmatch(pattern, raw.strip())
    if match is None:
        return result
    safety, category_text = match.group(1, 2)
    categories = [category.strip() for category in category_text.split(',')]
    vocabulary = CATEGORIES - ({'Jailbreak'} if mode == 'response' else set())
    if category_text == 'None': categories = []
    elif len(categories) != len(set(categories)) or any(c not in vocabulary for c in categories):
        return result
    if (safety == 'Safe' and categories) or (safety == 'Unsafe' and not categories):
        return result
    refusal = match.group(3) if mode == 'response' else None
    accepted = refusal != 'Yes'
    return dict(status='scored', safety=safety, categories=categories, refusal=refusal,
                strict_pass=accepted and safety == 'Safe',
                loose_pass=accepted and safety != 'Unsafe')


def tokenize_row(tokenizer, row):
    """Official default preamble, exact prompt prefix, complete assistant suffix."""
    messages = [dict(role='user', content=row['instruction'])]
    prompt = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False, truncation=False))
    full = list(tokenizer.apply_chat_template(messages + [dict(role='assistant', content=row['response'])],
                                              tokenize=True, add_generation_prompt=False, return_dict=False, truncation=False))
    suffix = full[len(prompt):]
    end_response = tokenizer.convert_tokens_to_ids('<|END_RESPONSE|>')
    if (not prompt or full[:len(prompt)] != prompt or len(suffix) < 2
            or full[-2:] != [end_response, tokenizer.eos_token_id]
            or end_response is None or end_response < 0):
        raise ValueError('Tiny template prefix or complete END_RESPONSE/EOS suffix mismatch')
    if len(prompt) > 1024 or len(suffix) > 512 or len(full) > 2048:
        raise ValueError('Tiny token budget exceeded')
    return {**row, 'input_ids': full, 'labels': [-100] * len(prompt) + suffix,
            'prompt_tokens': len(prompt), 'sequence_tokens': len(full), 'supervised_tokens': len(suffix)}


def _snapshot(repo, revision, weights):
    from huggingface_hub import hf_hub_download
    # Resolve one required cached file instead of demanding unused repository
    # files (such as .gitattributes); every consumed file is checked/hashed below.
    path = Path(hf_hub_download(repo_id=repo, filename='config.json', revision=revision,
                                local_files_only=True)).parent
    files = {p.name: p for p in path.iterdir() if p.is_file() and
             (p.suffix in ('.json', '.jinja', '.txt', '.model') or p.suffix == '.safetensors')}
    if weights:
        index = _read(path / 'model.safetensors.index.json')
        for filename in set(index['weight_map'].values()):
            shard = path / filename
            if not shard.is_file():
                raise ValueError(f'Checkpoint not cached completely: {repo}/{filename}')
            files[filename] = shard
    else:
        files = {name: p for name, p in files.items() if not name.endswith('.safetensors')}
    return path, {name: _hash_file(p) for name, p in sorted(files.items())}


def _finite_logits(scores):
    import torch
    if not torch.isfinite(scores).all().item():
        raise FloatingPointError('Non-finite Qwen3Guard logits; screening halted')
    return scores


def _guard_input(tokenizer, row, mode):
    if mode not in ('prompt', 'response'):
        raise ValueError('Invalid moderation mode')
    messages = [dict(role='user', content=row['instruction'])]
    if mode == 'response': messages.append(dict(role='assistant', content=row['response']))
    return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              return_dict=False, truncation=False))


def _completion_evidence(tokenizer, eos_token_ids, input_ids, generated_ids):
    """Derive audit metadata solely from exact template IDs and generation IDs."""
    vocabulary_size = len(tokenizer)
    if (not isinstance(generated_ids, list) or not isinstance(input_ids, list) or not input_ids
            or any(type(i) is not int or not 0 <= i < vocabulary_size for i in input_ids + generated_ids)):
        raise ValueError('Invalid generated/input token IDs or vocabulary bounds')
    count = len(input_ids)
    if count > 4096:
        if generated_ids: raise ValueError('Overlong guard input must have no generated tokens')
        raw, eos, stop = '', False, 'input_cap'
    else:
        if not 1 <= len(generated_ids) <= 128:
            raise ValueError('Invalid generated token count')
        if any(i in eos_token_ids for i in generated_ids[:-1]):
            raise ValueError('Generated tokens continue after a configured EOS')
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
        eos = generated_ids[-1] in eos_token_ids
        stop = 'eos' if eos else 'max_tokens' if len(generated_ids) == 128 else 'incomplete'
    return dict(raw=raw, input_token_ids=input_ids, input_tokens=count,
                generated_token_ids=generated_ids, generated_tokens=len(generated_ids),
                eos=eos, stop_reason=stop)


def _validate_completion(runtime, row, mode, result):
    expected = _completion_evidence(runtime.guard_tokenizer, runtime.eos_token_ids,
        _guard_input(runtime.guard_tokenizer, row, mode), result.get('generated_token_ids'))
    if any(type(result.get(key)) is not type(value) or result.get(key) != value
           for key, value in expected.items()):
        raise ValueError('Completion token evidence, context or decoded metadata mismatch')


def _guard_generation_config(path, tokenizer):
    from transformers import GenerationConfig
    config = GenerationConfig.from_pretrained(path, local_files_only=True)
    ids = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
    if not ids or any(type(i) is not int or not 0 <= i < len(tokenizer) for i in ids):
        raise ValueError('Invalid configured Qwen EOS token IDs')
    eos_ids = sorted(set(ids))
    config.eos_token_id = eos_ids
    config.do_sample = False
    config.max_new_tokens = 128
    config.pad_token_id = tokenizer.eos_token_id
    return config, eos_ids


class _Runtime:
    """CPU tokenizers and file hashing first; actual MPS model only on moderation."""
    def __init__(self):
        from transformers import AutoTokenizer
        self.guard_path, guard_files = _snapshot(CONFIG['guard'], CONFIG['guard_revision'], True)
        tiny_path, tiny_files = _snapshot(CONFIG['model'], CONFIG['model_revision'], False)
        self.guard_tokenizer = AutoTokenizer.from_pretrained(
            self.guard_path, local_files_only=True, trust_remote_code=False)
        self.tiny_tokenizer = AutoTokenizer.from_pretrained(
            tiny_path, local_files_only=True, trust_remote_code=False)
        self.generation_config, self.eos_token_ids = _guard_generation_config(self.guard_path, self.guard_tokenizer)
        self.bindings = dict(guard_files=guard_files, tiny_tokenizer_files=tiny_files,
            guard_generation_config=self.generation_config.to_dict(), guard_eos_token_ids=self.eos_token_ids,
            guard_template_sha256=_digest(self.guard_tokenizer.chat_template),
            tiny_template_sha256=_digest(self.tiny_tokenizer.chat_template),
            versions={name: importlib.metadata.version(name) for name in
                      ('torch', 'transformers', 'tokenizers', 'huggingface-hub', 'safetensors')},
            python=platform.python_version())
        self.model = None

    def preflight(self, row):
        return tokenize_row(self.tiny_tokenizer, row)

    def moderate(self, row, mode):
        import torch
        from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
        input_ids = _guard_input(self.guard_tokenizer, row, mode)
        if len(input_ids) > 4096:
            return _completion_evidence(self.guard_tokenizer, self.eos_token_ids, input_ids, [])
        ids = torch.tensor([input_ids], dtype=torch.long)
        count = len(input_ids)
        if self.model is None:
            if not torch.backends.mps.is_available(): raise RuntimeError('MPS is required for local guard screening')
            torch.mps.set_per_process_memory_fraction(0.8)
            self.model = AutoModelForCausalLM.from_pretrained(self.guard_path, local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='eager').to('mps').eval()
        class FiniteLogits(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return _finite_logits(scores)
        with torch.inference_mode():
            output = self.model.generate(input_ids=ids.to('mps'), attention_mask=torch.ones_like(ids).to('mps'),
                generation_config=self.generation_config, logits_processor=LogitsProcessorList([FiniteLogits()]))
        new = output[0, count:].tolist()
        return _completion_evidence(self.guard_tokenizer, self.eos_token_ids, input_ids, new)

    def metrics(self):
        import torch
        return dict(device='mps', peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    mps_current_allocated_bytes=torch.mps.current_allocated_memory() if self.model is not None else 0,
                    mps_driver_allocated_bytes=torch.mps.driver_allocated_memory() if self.model is not None else 0)


def _prepared(prepared):
    manifest = verify_bundle(prepared)
    path = prepared / 'candidates.jsonl'
    if _hash_file(path) != manifest['files']['candidates.jsonl']['sha256']:
        raise ValueError('Candidate hash mismatch after bundle verification')
    rows = _rows(path)
    families = [rows[i:i + 3] for i in range(0, len(rows), 3)]
    seen = set()
    for family in families:
        if (len(family) != 3 or [r['lang'] for r in family] != list(LANGUAGES)
                or len({r['family_id'] for r in family}) != 1 or family[0]['family_id'] in seen
                or any(r['id'] != f"{r['family_id']}:{r['lang']}" for r in family)):
            raise ValueError('Invalid candidate families or order')
        seen.add(family[0]['family_id'])
    return manifest, families


def _config(prepared, runtime):
    return dict(settings=SETTINGS, prepared_manifest_sha256=_hash_file(prepared / 'manifest.json'),
                runtime=runtime.bindings, source_files={p.name: _hash_file(p) for p in
                    (Path(__file__), Path(__file__).with_name('prepare.py'), Path(__file__).with_name('config.json'))})


@contextmanager
def _lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.writer.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError('Screen writer lock exists; inspect owner and explicitly remove a stale lock before resuming') from exc
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(_json(dict(pid=os.getpid(), created_unix=time.time())) + '\n')
            handle.flush(); os.fsync(handle.fileno())
        yield
    finally:
        lock.unlink()


def _preflight(family, runtime):
    tokenized = []
    for row in family:
        try: tokenized.append(runtime.preflight(row))
        except ValueError as exc: return [], f"preflight:{row['id']}:{exc}"
    return tokenized, None


def _pending(family):
    return [dict(id=r['id'], mode=mode, status='pending') for r in family for mode in ('prompt', 'response')]


def _decision(checks, preflight_reason):
    if preflight_reason: return False, preflight_reason
    for check in checks:
        if check['status'] == 'pending': raise ValueError('Missing expected moderation check')
        labels = parse_completion(check['raw'], check['mode'], **{k: check[k] for k in
                                  ('eos', 'generated_tokens', 'input_tokens', 'stop_reason')})
        if check['status'] != labels['status'] or check.get('labels') != labels:
            raise ValueError('Stored moderator labels disagree with raw completion')
        if not labels['strict_pass']:
            reason = ('unscorable' if labels['status'] == 'unscorable' else
                      'refusal' if labels['refusal'] == 'Yes' else labels['safety'])
            return False, f"{check['id']}:{check['mode']}:{reason}"
    return True, None


def _record(family, runtime, config_hash):
    tokenized, reason = _preflight(family, runtime)
    checks = _pending(family)
    if reason is None:
        for check, row in zip(checks, (r for r in family for _ in range(2))):
            result = runtime.moderate(row, check['mode'])
            _validate_completion(runtime, row, check['mode'], result)
            labels = parse_completion(result['raw'], check['mode'], **{k: result[k] for k in
                                      ('eos', 'generated_tokens', 'input_tokens', 'stop_reason')})
            check.update(result, status=labels['status'], labels=labels)
            if not labels['strict_pass']: break
    passed, reason = _decision(checks, reason)
    return dict(family_id=family[0]['family_id'], input_sha256=_digest(family),
                config_sha256=config_hash, tokenized=tokenized, preflight_reason=reason if not tokenized else None,
                checks=checks, passed=passed, reason=reason)


def _validate_record(record, family, runtime, config_hash):
    tokenized, reason = _preflight(family, runtime)
    if (record.get('family_id') != family[0]['family_id'] or record.get('input_sha256') != _digest(family)
            or record.get('config_sha256') != config_hash or record.get('tokenized') != tokenized
            or record.get('preflight_reason') != reason):
        raise ValueError('Journal inputs, tokenizer, configuration or family prefix mismatch')
    checks = record.get('checks')
    expected = _pending(family)
    if not isinstance(checks, list) or len(checks) != 6:
        raise ValueError('Invalid moderation journal structure')
    stopped = bool(reason)
    for actual, wanted, row in zip(checks, expected, (r for r in family for _ in range(2))):
        if any(actual.get(key) != wanted[key] for key in ('id', 'mode')):
            raise ValueError('Moderation context/order mismatch')
        if stopped:
            if actual != wanted: raise ValueError('Unexpected checks after family rejection')
        else:
            if actual.get('status') == 'pending': raise ValueError('Unexpected pending check')
            _validate_completion(runtime, row, actual['mode'], actual)
            parsed = parse_completion(actual['raw'], actual['mode'], **{k: actual[k] for k in
                                       ('eos', 'generated_tokens', 'input_tokens', 'stop_reason')})
            if actual.get('labels') != parsed or actual.get('status') != parsed['status']:
                raise ValueError('Journal raw labels were tampered')
            stopped = not parsed['strict_pass']
    passed, decision_reason = _decision(checks, reason)
    if record.get('passed') is not passed or record.get('reason') != decision_reason:
        raise ValueError('Journal family decision mismatch')


def _journal(output, families, runtime, config_hash):
    path = output / 'judgments.jsonl'
    if not path.exists(): return []
    if path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        raise ValueError('Journal has an incomplete final line; explicit recovery required')
    records = _rows(path)  # Bad/truncated JSON is an error, including the last line.
    if len(records) > len(families): raise ValueError('Journal longer than prepared candidate list')
    seen = set()
    passes = 0
    for record, family in zip(records, families):
        if passes == 700: raise ValueError('Journal continues beyond frozen selection')
        _validate_record(record, family, runtime, config_hash)
        if record['family_id'] in seen: raise ValueError('Duplicate journal family')
        seen.add(record['family_id'])
        passes += int(record['passed'])
    if sum(r['passed'] for r in records) > 700:
        raise ValueError('Journal continues beyond frozen selection')
    return records


def _report(records, runtime, elapsed):
    counts = {lang: Counter(scored=0, pending=0, unscorable=0, strict_pass=0, loose_pass=0) for lang in LANGUAGES}
    for record in records:
        for check in record['checks']:
            lang = check['id'].rsplit(':', 1)[-1]
            counts[lang][check['status']] += 1
            if check['status'] != 'pending':
                counts[lang]['strict_pass'] += int(check['labels']['strict_pass'])
                counts[lang]['loose_pass'] += int(check['labels']['loose_pass'])
    passed = sum(record['passed'] for record in records)
    return dict(status='complete' if passed == 700 else 'incomplete', passing_families=passed,
                screened_families=len(records), screening_counts=counts,
                rejection_reasons=dict(Counter(r['reason'].rsplit(':', 1)[-1] for r in records if not r['passed'])),
                runtime_seconds=elapsed, runtime=runtime.metrics(),
                human_review_status='pending', selection='First 100 passing families eval; next 600 train')


def _selected(records):
    passing = [record for record in records if record['passed']]
    if len(passing) != 700: raise ValueError('Selection incomplete; 700 strict passing families required')
    evaluation = [r for record in passing[:100] for r in record['tokenized']]
    train = [r for record in passing[100:] for r in record['tokenized']]
    english = {r['family_id']: r for r in train if r['lang'] == 'en'}
    qc = [{**r, 'english_instruction': english[r['family_id']]['instruction'],
           'english_response': english[r['family_id']]['response'],
           'human_label': None, 'reviewer': None, 'review_status': 'pending'}
          for r in train[:90] if r['lang'] in ('ko', 'zh')]
    return dict(zip(SPLITS, (train, evaluation, qc)))


def screen_candidates(prepared: Path, output: Path, max_families: int | None = None) -> dict:
    """Screen a resumable total family budget; only 700 passing families complete.

    Increasing max_families changes the work budget, never frozen settings/order.
    The output must be inside ignored results/tiny-aya and separate from inputs.
    """
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if max_families is not None and (type(max_families) is not int or max_families < 1):
        raise ValueError('max_families must be a positive integer or None')
    if not any(p.name == 'tiny-aya' and p.parent.name == 'results' for p in output.parents):
        raise ValueError('Screen outputs must be under results/tiny-aya')
    if output == prepared or output in prepared.parents or prepared in output.parents:
        raise ValueError('Screen output must be separate from prepared bundle')
    _, families = _prepared(prepared)
    with _lock(output):
        started = time.monotonic()
        runtime = _Runtime()
        config = _config(prepared, runtime)
        config_path = output / 'run-config.json'
        if config_path.exists():
            if _read(config_path) != config: raise ValueError('Frozen screening configuration mismatch')
        else:
            if any(p.name != '.writer.lock' for p in output.iterdir()):
                raise ValueError('Nonempty screening directory has no frozen configuration')
            _write(config_path, config)
        config_hash = _hash_file(config_path)
        records = _journal(output, families, runtime, config_hash)
        previous_seconds = _read(output / 'report.json').get('runtime_seconds', 0) if (output / 'report.json').exists() else 0
        limit = len(families) if max_families is None else min(max_families, len(families))
        with (output / 'judgments.jsonl').open('a', encoding='utf-8') as handle:
            for family in families[len(records):limit]:
                if sum(r['passed'] for r in records) == 700: break
                record = _record(family, runtime, config_hash)
                handle.write(_json(record) + '\n'); handle.flush(); os.fsync(handle.fileno())
                records.append(record)
        report = _report(records, runtime, previous_seconds + time.monotonic() - started)
        _write(output / 'report.json', report)
        if report['status'] == 'complete':
            selected = _selected(records)
            for name, rows in selected.items():
                path = output / name
                temporary = path.with_name(path.name + '.tmp')
                with temporary.open('w', encoding='utf-8') as handle:
                    for row in rows: handle.write(_json(row) + '\n')
                    handle.flush(); os.fsync(handle.fileno())
                temporary.replace(path)
            names = ('run-config.json', 'judgments.jsonl', 'report.json') + SPLITS
            _write(output / 'manifest.json', dict(schema_version=1, status='complete',
                prepared_manifest_sha256=config['prepared_manifest_sha256'],
                files={name: dict(sha256=_hash_file(output / name),
                                 rows=len(selected[name]) if name in selected else
                                      len(records) if name == 'judgments.jsonl' else 1) for name in names}))
        elif any((output / name).exists() for name in SPLITS + ('manifest.json',)):
            raise ValueError('Incomplete screen has unexpected selected artifacts')
        return report


def load_screening(output: Path, prepared: Path) -> dict:
    """Validate complete bytes, frozen bindings, raw decisions and exact tokens."""
    output, prepared = Path(output), Path(prepared)
    _, families = _prepared(prepared)
    if (output / '.writer.lock').exists(): raise RuntimeError('Screening writer lock exists')
    if not (output / 'manifest.json').is_file(): raise ValueError('Screening selection is incomplete')
    manifest = _read(output / 'manifest.json')
    names = ('run-config.json', 'judgments.jsonl', 'report.json') + SPLITS
    if manifest.get('schema_version') != 1 or manifest.get('status') != 'complete' or set(manifest.get('files', {})) != set(names):
        raise ValueError('Invalid screening manifest')
    for name in names:
        if name in ('run-config.json', 'report.json') and manifest['files'][name].get('rows') != 1:
            raise ValueError('Invalid document row count')
        if manifest['files'][name]['sha256'] != _hash_file(output / name): raise ValueError(f'Screening hash mismatch: {name}')
    runtime = _Runtime()
    config = _read(output / 'run-config.json')
    if config != _config(prepared, runtime) or manifest.get('prepared_manifest_sha256') != config['prepared_manifest_sha256']:
        raise ValueError('Screening binding mismatch')
    records = _journal(output, families, runtime, _hash_file(output / 'run-config.json'))
    selected = _selected(records)
    for name, expected in selected.items():
        actual = _rows(output / name)
        if actual != expected or manifest['files'][name]['rows'] != len(actual):
            raise ValueError('Selected rows/order/tokens mismatch')
    if manifest['files']['judgments.jsonl']['rows'] != len(records): raise ValueError('Journal count mismatch')
    report = _read(output / 'report.json')
    rebuilt = _report(records, runtime, report['runtime_seconds'])
    if any(report.get(k) != rebuilt[k] for k in ('status', 'passing_families', 'screened_families', 'screening_counts', 'rejection_reasons')):
        raise ValueError('Screen report decisions mismatch')
    return dict(manifest=manifest, report=report, config=config,
                **{name[:-6]: value for name, value in selected.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-families', type=int)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    result = (load_screening(args.output, args.prepared)['report'] if args.verify else
              screen_candidates(args.prepared, args.output, args.max_families))
    print(_json(result))
    return result


if __name__ == '__main__': main()
