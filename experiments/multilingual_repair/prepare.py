"""Freeze public instruction and multilingual safety partitions without generation.

The task pool is a new split of the public Aya *training* partition. It is held
out from this experiment's optimizer, not certified absent from base pretraining.
MultiJail translations are assigned together by source ID. No conversational or
private project data enters this experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import unicodedata

SOURCES = {
    'aya_dataset': {'repo': 'CohereLabs/aya_dataset',
        'revision': 'f9ea04583f02a8f86404ff6c58bf75fe637df8a2',
        'filename': 'data/train-00000-of-00001.parquet', 'license': 'Apache-2.0'},
    'multijail': {'repo': 'DAMO-NLP-SG/MultiJail',
        'revision': '93d65c778925d55973b9ff30c7bccc8545494eb6',
        'filename': 'MultiJail.csv', 'license': 'MIT'},
}
LANGUAGES = {'eng': 'en', 'kor': 'ko', 'zho': 'zh'}
SYSTEMS = {
    'en': 'You are a helpful multilingual assistant. Answer in English. Be accurate, follow the requested format, and do not invent facts.',
    'ko': '당신은 유용한 다국어 도우미입니다. 한국어로 답하세요. 정확하게 답하고 요청된 형식을 따르며 사실을 지어내지 마세요.',
    'zh': '你是一位乐于助人的多语言助手。请用中文回答。回答要准确，遵循要求的格式，不要编造事实。',
}
# A conservative lexical screen, not a validated semantic safety classifier.
RISK = re.compile(r'\b(kill|murder|suicide|bomb|weapon|hack|hacking|phish|porn|rape|fraud|cocaine|methamphetamine|terrorist|nazi|sex|sexual)\b|살인|자살|폭탄|마약|성폭행|해킹|杀人|自杀|炸弹|毒品|强奸|色情|诈骗', re.I)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def rank(seed, identity):
    return sha_bytes(f'{seed}:{identity}'.encode())


def prepare_tokens(row, tokenizer, max_tokens):
    """Preserve the complete prompt and supervise only a complete target + EOS."""
    messages = [{'role': 'system', 'content': row['system']}, {'role': 'user', 'content': row['request']}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    full = prefix
    if 'reference' in row:
        full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': row['reference']}],
                                             tokenize=True, add_generation_prompt=False, return_dict=False)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError('Assistant prefix mismatch or empty supervision')
        if full[-1] != tokenizer.eos_token_id:
            raise ValueError('Missing terminal EOS')
        if tokenizer.decode(full[len(prefix):-1], skip_special_tokens=False) != row['reference']:
            raise ValueError('Assistant target tokenization does not round-trip exactly')
    if len(full) > max_tokens:
        raise ValueError('Oversized sequence; truncation is forbidden')
    result = dict(row, input_ids=full, attention_mask=[1] * len(full), prompt_tokens=len(prefix), sequence_tokens=len(full))
    if 'reference' in row:
        result.update(labels=[-100] * len(prefix) + full[len(prefix):], supervised_tokens=len(full)-len(prefix))
    return result


def partition_tasks(rows, config):
    """Split unique normalized requests globally; confirmation selected before train."""
    unique = {}
    for row in sorted(rows, key=lambda r: r['source_id']):
        unique.setdefault(key(row['request']), row)
    by_lang = defaultdict(list)
    for row in unique.values():
        by_lang[row['language']].append(row)
    partitions = {name: [] for name in ('train', 'calibration', 'development', 'confirmation')}
    counts = config['data']
    for lang in ('en', 'ko', 'zh'):
        ordered = sorted(by_lang[lang], key=lambda r: rank(config['seed'], r['source_id']))
        sizes = [('confirmation', counts['confirmation_per_language']),
                 ('development', counts['development_per_language']),
                 ('calibration', counts['calibration_per_language']), ('train', counts['train_counts'][lang])]
        if len(ordered) < sum(n for _, n in sizes):
            raise ValueError(f'Insufficient unique eligible {lang} rows: {len(ordered)}')
        at = 0
        for split, n in sizes:
            partitions[split].extend(dict(r, split=split) for r in ordered[at:at+n])
            at += n
    assert_disjoint(partitions)
    return partitions


def assert_disjoint(partitions):
    ids, prompts, families = {}, {}, {}
    for split, rows in partitions.items():
        for row in rows:
            for value, seen, label in [(row['id'], ids, 'id'), (key(row['request']), prompts, 'prompt')]:
                if value in seen:
                    raise ValueError(f'Duplicate {label} in {split} and {seen[value]}')
                seen[value] = split
            family = row.get('family_id')
            if family is not None:
                if family in families and families[family] != split:
                    raise ValueError('Translated safety family crosses partitions')
                families[family] = split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Use a new preparation directory')
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from curatorkit.normalizers import TextCleaner, ExactDeduplicator
    from curatorkit.schema import DataSample

    config = json.loads(args.config.read_text())
    tok = AutoTokenizer.from_pretrained(config['model'], revision=config['model_revision'], local_files_only=True, trust_remote_code=False)
    paths, sources = {}, {}
    for name, source in SOURCES.items():
        paths[name] = Path(hf_hub_download(source['repo'], source['filename'], repo_type='dataset', revision=source['revision']))
        sources[name] = dict(source, source_sha256=sha_bytes(paths[name].read_bytes()))
    selected = []
    stats = Counter()
    for index, raw in enumerate(pq.read_table(paths['aya_dataset'], columns=['inputs', 'targets', 'language_code']).to_pylist()):
        lang = LANGUAGES.get(raw['language_code'])
        if lang is None:
            continue
        stats[f'{lang}:source'] += 1
        request, reference = raw['inputs'].strip(), raw['targets'].strip()
        if not request or not reference:
            stats[f'{lang}:empty'] += 1; continue
        if RISK.search(request + '\n' + reference):
            stats[f'{lang}:lexical_risk'] += 1; continue
        if any(s in request + reference for s in tok.all_special_tokens):
            stats[f'{lang}:special_token_text'] += 1; continue
        prompt_n = len(tok(request, add_special_tokens=False)['input_ids'])
        target_n = len(tok(reference, add_special_tokens=False)['input_ids'])
        if not (4 <= prompt_n <= config['data']['max_request_tokens'] and 4 <= target_n <= config['data']['max_reference_tokens']):
            stats[f'{lang}:length_filter'] += 1; continue
        sid = f'aya-train-{index:06d}'
        selected.append({'id': sid, 'source_id': sid, 'language': lang, 'category': 'task',
                         'request': request, 'reference': reference, 'system': SYSTEMS[lang]})

    # CuratorKIT is executed on public requests; references retain their original text.
    transforms = {'strip_html': False, 'normalise_unicode': True, 'fix_encoding_artifacts': False,
                  'collapse_whitespace': False, 'remove_control_chars': True}
    samples = [DataSample(id=r['id'], source_uri='hf://datasets/' + SOURCES['aya_dataset']['repo'],
                         task_type='prompt_only', instruction=r['request']) for r in selected]
    cleaned = TextCleaner(transforms=transforms, fields=['instruction']).run(samples)
    kept = ExactDeduplicator().run(cleaned)
    originals = {r['id']: r for r in selected}
    eligible = []
    changed = 0
    for sample in kept:
        row = dict(originals[sample.id], request=sample.instruction)
        changed += row['request'] != originals[sample.id]['request']
        try:
            eligible.append(prepare_tokens(row, tok, config['training']['max_sequence_tokens']))
        except ValueError:
            stats[f"{row['language']}:tokenization_rejected"] += 1
    partitions = partition_tasks(eligible, config)

    with paths['multijail'].open(encoding='utf-8-sig', newline='') as handle:
        raw_safety = list(csv.DictReader(handle))
    family_rows = []
    seen_safety_prompts = set()
    for raw in sorted(raw_safety, key=lambda r: rank(config['seed'], 'multijail:' + r['id'])):
        family = []
        for lang in ('en', 'ko', 'zh'):
            request = raw[lang].strip()
            if not request or key(request) in seen_safety_prompts or any(s in request for s in tok.all_special_tokens):
                family = []; break
            row = {'id': f"multijail-{raw['id']}-{lang}", 'source_id': raw['id'],
                   'family_id': f"multijail-{raw['id']}", 'language': lang, 'category': 'disallowed',
                   'request': request, 'system': SYSTEMS[lang]}
            try:
                family.append(prepare_tokens(row, tok, config['training']['max_sequence_tokens']))
            except ValueError:
                family = []; break
        if len(family) == 3:
            family_rows.append(family)
            seen_safety_prompts.update(key(r['request']) for r in family)
    sizes = [('confirmation', config['data']['safety_confirmation_families']),
             ('development', config['data']['safety_development_families']),
             ('alignment', config['data']['alignment_families'])]
    if len(family_rows) < sum(n for _, n in sizes):
        raise ValueError('Insufficient complete MultiJail families')
    partitions['alignment'] = []
    at = 0
    for split, n in sizes:
        partitions[split].extend(dict(r, split=split) for family in family_rows[at:at+n] for r in family)
        at += n
    assert_disjoint(partitions)
    args.output.mkdir(parents=True)
    (args.output / 'config.json').write_bytes(args.config.read_bytes())
    for split, rows in partitions.items():
        with (args.output / f'{split}.jsonl').open('x') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    manifest = {
        'schema_version': 1, 'status': 'passed_tokenization_preflight', 'sources': sources,
        'files': {p.name: sha_bytes(p.read_bytes()) for p in sorted(args.output.iterdir()) if p.is_file()},
        'preparation_script_sha256': sha_bytes(Path(__file__).read_bytes()),
        'tokenizer': {'model': config['model'], 'actual_snapshot_commit': config['model_revision'],
                      'eos_token_id': tok.eos_token_id, 'chat_template_sha256': sha_bytes(tok.get_chat_template().encode())},
        'counts': {s: dict(Counter(r['language'] + ':' + r['category'] for r in rows)) for s, rows in partitions.items()},
        'max_sequence_tokens': {s: max(r['sequence_tokens'] for r in rows) for s, rows in partitions.items()},
        'curation': {'tool': 'CuratorKIT', 'version': importlib.metadata.version('curatorkit'),
                     'cleaning_transforms': transforms, 'input_rows': len(samples), 'deduplicated_rows': len(kept),
                     'changed_requests': changed, 'filter_counts': dict(stats), 'risk_filter': RISK.pattern},
        'truncated_examples': 0, 'optimizer_steps': 0, 'model_generations': 0,
        'limitations': ['Task splits are newly held out from this fine-tuning, not from original base pretraining.',
                       'Task data is lexically screened, not fully reviewed as harmless or correct.',
                       'Korean training count is lower because the public source has fewer eligible Korean rows.',
                       'Task exact duplicates are removed globally; semantic/translated paraphrase independence is not certified.',
                       'MultiJail labels come from its source; no semantic outcome labels are inferred from refusal words.',
                       'Prompt-only alignment KL differs from alignment prompt/response protocols; this is a scoped adaptation.'],
    }
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'counts': manifest['counts'], 'max_sequence_tokens': manifest['max_sequence_tokens']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
