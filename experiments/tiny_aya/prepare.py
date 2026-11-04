"""Deterministic CPU preparation of pinned public multilingual source data.

No model or tokenizer is loaded. Candidate references await Qwen screening;
source translations await human review. Lexical deduplication is a heuristic,
not evidence of semantic independence or absence from model pretraining.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
from math import ceil
from pathlib import Path
import re
import unicodedata

CONFIG_PATH = Path(__file__).with_name('config.json')
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
LANGUAGES = ('en', 'ko', 'zh')
ATTACK_LANGUAGES = ('en', 'ko', 'zh', 'sw')
DATA_FILES = ('candidates.jsonl', 'attacks.jsonl', 'alignment.jsonl', 'translation_review.jsonl')
REFUSAL = re.compile(r'as\s+an\s+ai|죄송|抱歉|作为\s*AI', re.IGNORECASE)
CONTROL_TOKEN = re.compile(
    r'<\|[^<>\n]+\|>|</?s>|<(?:pad|bos|eos)(?:_token)?>|\[/?INST\]|'
    r'<<\/?SYS>>|<(?:start|end)_of_turn>|<turn_sep>', re.IGNORECASE)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _rank(identity: str) -> str:
    return _sha(f"{CONFIG['seed']}:{identity}".encode('utf-8'))


def _identity(value) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    value = str(value).strip()
    return value or None


def _valid_text(text) -> bool:
    return (isinstance(text, str) and len(''.join(text.split())) >= 4
            and REFUSAL.search(text) is None and CONTROL_TOKEN.search(text) is None)


def _normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _grams(text: str) -> frozenset[str]:
    text = _normalized(text)
    return frozenset(text[i:i + 5] for i in range(max(1, len(text) - 4)))


class _NearIndex:
    """Lossless prefix candidate filtering followed by exact Jaccard checks.

    A common total ordering by source gram frequency makes shared prefixes a
    necessary condition for Jaccard >= threshold. Only retained instructions
    enter the inverted index; no all-source pairwise scan is performed.
    """
    def __init__(self, all_grams, threshold=0.85):
        self.threshold = threshold
        self.frequency = Counter(g for grams in all_grams for g in grams)
        self.postings = defaultdict(set)
        self.values = []

    def _prefix(self, grams):
        ordered = sorted(grams, key=lambda gram: (self.frequency[gram], gram))
        return ordered[:len(ordered) - ceil(self.threshold * len(ordered)) + 1]

    def matches(self, grams):
        candidates = set()
        for gram in self._prefix(grams):
            candidates.update(self.postings.get(gram, ()))
        for index in sorted(candidates):
            previous = self.values[index]
            if not self.threshold * len(grams) <= len(previous) <= len(grams) / self.threshold:
                continue
            overlap = len(grams & previous)
            if overlap / (len(grams) + len(previous) - overlap) >= self.threshold:
                return True
        return False

    def add(self, grams):
        index = len(self.values)
        self.values.append(grams)
        for gram in self._prefix(grams):
            self.postings[gram].add(index)


def _provenance(sample, original):
    steps = []
    for record in sample.provenance_chain:
        step = record.model_dump(mode='json')
        # Curator timestamps reflect execution time, not source provenance.
        step.pop('timestamp', None)
        steps.append(step)
    return {'original_instruction': original, 'instruction_changed': sample.instruction != original,
            'steps': steps}


def select_candidates(data: dict, count: int = 1000) -> tuple[list[dict], dict]:
    """Filter complete Bactrian ID families, clean instructions and rank by hash."""
    if type(count) is not int or count < 1:
        raise ValueError('candidate_families must be a positive integer')
    from curatorkit.normalizers import ExactDeduplicator, TextCleaner
    from curatorkit.schema import DataSample

    by_language = {}
    for lang in LANGUAGES:
        grouped = defaultdict(list)
        for row in data.get(lang, []):
            identity = _identity(row.get('id'))
            if identity is not None:
                grouped[identity].append(row)
        by_language[lang] = grouped
    identities = set().union(*(set(rows) for rows in by_language.values()))
    stats = Counter(source_families=len(identities), rejected_incomplete_families=0,
                    rejected_content_families=0, rejected_duplicate_id_families=0,
                    rejected_exact_duplicate_families=0, rejected_near_duplicate_families=0,
                    cleaner_changed_instructions=0, selected_changed_instructions=0)
    samples, originals, references, families = [], {}, {}, []
    source = CONFIG['sources']['bactrian']
    for identity in sorted(identities, key=lambda i: (_rank(i), i)):
        raw = [by_language[lang].get(identity, []) for lang in LANGUAGES]
        if any(len(rows) > 1 for rows in raw):
            stats['rejected_duplicate_id_families'] += 1
            continue
        if any(len(rows) != 1 for rows in raw):
            stats['rejected_incomplete_families'] += 1
            continue
        rows = [values[0] for values in raw]
        if any(not isinstance(row.get('input'), str) or row['input'].strip()
               or not _valid_text(row.get('instruction')) or not _valid_text(row.get('output'))
               for row in rows):
            stats['rejected_content_families'] += 1
            continue
        family = []
        for lang, row in zip(LANGUAGES, rows):
            sid = f'bactrian:{identity}:{lang}'
            original = row['instruction'].strip()
            originals[sid] = original
            references[sid] = row['output'].strip()
            sample = DataSample(id=sid, source_uri='hf://datasets/' + source['repo'],
                                task_type='prompt_only', instruction=original)
            samples.append(sample)
            family.append(sample)
        families.append((identity, family))
    cleaned = TextCleaner(transforms=CONFIG['curator']['transforms'],
                          fields=CONFIG['curator']['fields']).run(samples)
    stats['cleaner_changed_instructions'] = sum(s.instruction != originals[s.id] for s in cleaned)
    kept_ids = {s.id for s in ExactDeduplicator(
        ignore_for_dedup=CONFIG['curator']['exact_ignore_for_dedup']).run(cleaned)}
    grams = {s.id: _grams(s.instruction) for s in cleaned}
    index = _NearIndex(grams.values())
    eligible = []
    for identity, family in families:
        if any(not _valid_text(s.instruction) for s in family):
            stats['rejected_content_families'] += 1
            continue
        if any(s.id not in kept_ids for s in family):
            stats['rejected_exact_duplicate_families'] += 1
            continue
        if any(index.matches(grams[s.id]) for s in family):
            stats['rejected_near_duplicate_families'] += 1
            continue
        eligible.append((identity, family))
        for sample in family:
            index.add(grams[sample.id])
    stats['eligible_families'] = len(eligible)
    if len(eligible) < count:
        raise ValueError(f'Insufficient eligible Bactrian families: {len(eligible)} available, {count} required')
    selected = []
    for identity, family in eligible[:count]:
        for lang, sample in zip(LANGUAGES, family):
            provenance = _provenance(sample, originals[sample.id])
            stats['selected_changed_instructions'] += provenance['instruction_changed']
            selected.append({'id': sample.id, 'family_id': f'bactrian:{identity}', 'lang': lang,
                             'instruction': sample.instruction, 'response': references[sample.id],
                             'source': source['repo'], 'source_revision': source['revision'],
                             'provenance': provenance})
    stats['selected_families'] = count
    return selected, dict(stats)


def _tags(value, identity):
    try:
        tags = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError(f'Malformed MultiJail tags for ID {identity}') from exc
    if not isinstance(tags, (list, tuple)) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise ValueError(f'Malformed MultiJail tags for ID {identity}: expected string list')
    return list(tags)


def select_multijail(rows: list[dict]) -> tuple[list, list, list, dict]:
    """Select source attacks and disjoint alignment prompts; generate no answers."""
    source = CONFIG['sources']['multijail']
    grouped = defaultdict(list)
    for row in rows:
        identity = _identity(row.get('id'))
        if identity is not None:
            grouped[identity].append(row)
    complete = []
    for identity in sorted(grouped, key=lambda i: (_rank(i), i)):
        values = grouped[identity]
        if len(values) != 1:
            continue
        row = values[0]
        tags = _tags(row.get('tags'), identity)
        if (not isinstance(row.get('source'), str) or not row['source'].strip()
                or any(not isinstance(row.get(lang), str) or not row[lang].strip() for lang in ATTACK_LANGUAGES)):
            continue
        complete.append((identity, row, tags))
    attack_count = CONFIG['selection']['attack_families']
    alignment_count = CONFIG['selection']['alignment_families']
    required = attack_count + alignment_count
    if len(complete) < required:
        raise ValueError(f'Insufficient complete MultiJail families: {len(complete)} available, {required} required')
    attacks, alignment, review = [], [], []
    counts = {}
    for name, selected, langs, output in (
        ('attack', complete[:attack_count], ATTACK_LANGUAGES, attacks),
        ('alignment', complete[attack_count:required], LANGUAGES, alignment),
    ):
        categories = Counter()
        for identity, raw, tags in selected:
            categories.update(set(tags))
            for lang in langs:
                output.append({'id': f'multijail:{identity}:{lang}', 'family_id': f'multijail:{identity}',
                               'lang': lang, 'instruction': raw[lang].strip(), 'source': raw['source'],
                               'source_revision': source['revision'], 'harm_categories': tags,
                               'source_dataset': source['repo']})
        counts[f'{name}_category_family_counts'] = dict(sorted(categories.items()))
    review_count = CONFIG['selection']['translation_review_families']
    for identity, raw, tags in complete[:review_count]:
        for lang in ('ko', 'zh'):
            review.append({'id': f'multijail:{identity}:{lang}:translation-review',
                           'family_id': f'multijail:{identity}', 'lang': lang,
                           'english_instruction': raw['en'].strip(), 'translated_instruction': raw[lang].strip(),
                           'source': raw['source'], 'source_dataset': source['repo'],
                           'source_revision': source['revision'], 'harm_categories': tags,
                           'review_status': 'pending', 'human_label': None, 'reviewer': None})
    return attacks, alignment, review, {
        'complete_families': len(complete), 'attack_families': attack_count,
        'alignment_families': alignment_count, 'translation_review_families': review_count, **counts}


def _check_output(output):
    output = Path(output)
    resolved = output.resolve()
    if not any(parent.name == 'tiny-aya' and parent.parent.name == 'results' for parent in resolved.parents):
        raise ValueError('Output must be a new directory beneath results/tiny-aya')
    if output.exists():
        raise FileExistsError('Refusing overwrite; choose a new preparation directory')
    return output


def prepare_sources(output: Path, candidate_families: int = 1000) -> dict:
    """Download pinned HF source bytes into its cache and prepare an ignored bundle."""
    output = _check_output(output)
    if type(candidate_families) is not int or candidate_families < 1:
        raise ValueError('candidate_families must be a positive integer')
    from huggingface_hub import hf_hub_download
    paths = {}
    source = CONFIG['sources']['bactrian']
    for lang, filename in source['files'].items():
        paths[lang] = Path(hf_hub_download(repo_id=source['repo'], filename=filename,
                                         repo_type='dataset', revision=source['revision']))
    source = CONFIG['sources']['multijail']
    paths['multijail'] = Path(hf_hub_download(repo_id=source['repo'], filename=source['filename'],
                                           repo_type='dataset', revision=source['revision']))
    return prepare_from_files(output, paths, candidate_families)


def prepare_from_files(output: Path, source_paths: dict, candidate_families: int = 1000) -> dict:
    """Prepare pinned-source bytes already in local files, including CPU fixtures."""
    output = _check_output(output)
    if type(candidate_families) is not int or candidate_families < 1:
        raise ValueError('candidate_families must be a positive integer')
    data, sources = {}, {}
    for lang in LANGUAGES:
        path = Path(source_paths[lang])
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            data[lang] = json.load(handle)
        source = CONFIG['sources']['bactrian']
        sources[lang] = {'repo': source['repo'], 'revision': source['revision'],
                         'filename': source['files'][lang], 'sha256': _sha(path.read_bytes())}
    path = Path(source_paths['multijail'])
    with path.open(encoding='utf-8-sig', newline='') as handle:
        safety = list(csv.DictReader(handle))
    source = CONFIG['sources']['multijail']
    sources['multijail'] = {'repo': source['repo'], 'revision': source['revision'],
                            'filename': source['filename'], 'sha256': _sha(path.read_bytes())}
    candidates, candidate_stats = select_candidates(data, candidate_families)
    attacks, alignment, review, safety_stats = select_multijail(safety)
    files = dict(zip(DATA_FILES, (candidates, attacks, alignment, review)))
    serialized = {name: ''.join(_json(row) + '\n' for row in rows).encode('utf-8') for name, rows in files.items()}
    config_bytes = (_json(CONFIG) + '\n').encode('utf-8')
    manifest = {'schema_version': 1, 'status': CONFIG['status'], 'config': CONFIG,
                'config_sha256': _sha(config_bytes), 'candidate_families': candidate_families,
                'source_files': sources, 'counts': {name[:-6]: len(rows) for name, rows in files.items()},
                'candidate_statistics': candidate_stats, 'multijail_statistics': safety_stats,
                'curation': {'tool': 'CuratorKIT', 'version': importlib.metadata.version('curatorkit'),
                             **CONFIG['curator'], 'provenance_timestamps': 'omitted for deterministic regeneration'},
                'files': {name: {'sha256': _sha(blob), 'rows': len(files[name])} for name, blob in serialized.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for name, blob in serialized.items():
        (output / name).write_bytes(blob)
    (output / 'config.json').write_bytes(config_bytes)
    (output / 'manifest.json').write_text(_json(manifest) + '\n', encoding='utf-8')
    return verify_bundle(output)


def _families(rows, languages, expected, dataset):
    grouped = defaultdict(list)
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in
                                             ('id', 'family_id', 'lang', 'instruction', 'source', 'source_revision')):
            raise ValueError('Invalid row structure')
        if row['id'] in ids or row['id'] != f"{row['family_id']}:{row['lang']}":
            raise ValueError('Duplicate or inconsistent row identity structure')
        ids.add(row['id'])
        source = CONFIG['sources'][dataset]
        if row['source_revision'] != source['revision']:
            raise ValueError('Incorrect source revision')
        if dataset == 'bactrian':
            if row['source'] != source['repo'] or not _valid_text(row.get('response')) or not _valid_text(row['instruction']):
                raise ValueError('Invalid candidate content/source structure')
            provenance = row.get('provenance')
            if not isinstance(provenance, dict) or not isinstance(provenance.get('original_instruction'), str):
                raise ValueError('Missing candidate provenance structure')
            original = provenance['original_instruction']
            steps = provenance.get('steps')
            if (type(provenance.get('instruction_changed')) is not bool
                    or provenance['instruction_changed'] != (original != row['instruction'])
                    or not isinstance(steps, list) or len(steps) != 2
                    or [step.get('step_name') for step in steps] != ['TextCleaner', 'ExactDeduplicator']
                    or any('timestamp' in step or not all(step.get(k) for k in ('step_version', 'config_hash'))
                           for step in steps)):
                raise ValueError('Invalid candidate provenance structure')
        elif (row.get('source_dataset') != source['repo']
              or not isinstance(row.get('harm_categories'), list)
              or any(not isinstance(tag, str) or not tag.strip() for tag in row['harm_categories'])
              or 'response' in row):
            raise ValueError('Invalid MultiJail source/category structure')
        grouped[row['family_id']].append(row)
    if len(grouped) != expected or len(rows) != expected * len(languages):
        raise ValueError('Incorrect family structure/count')
    for family in grouped.values():
        if [row['lang'] for row in family] != list(languages):
            raise ValueError('Invalid language structure')
        if dataset == 'multijail' and any(
                any(row[key] != family[0][key] for key in ('source', 'source_dataset', 'source_revision', 'harm_categories'))
                for row in family):
            raise ValueError('Inconsistent MultiJail family source/category metadata')
    return grouped


def verify_bundle(path: Path) -> dict:
    """Verify byte digests, frozen pins, full families, partitions and review pairs.

    The manifest is a local integrity record, not a cryptographic signature of
    remote source authenticity. Source hashes refer to the pinned downloaded bytes.
    """
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('config') != CONFIG:
        raise ValueError('Configuration/source pins do not match this preparation version')
    config_bytes = (path / 'config.json').read_bytes()
    if _sha(config_bytes) != manifest.get('config_sha256') or json.loads(config_bytes) != CONFIG:
        raise ValueError('Configuration hash mismatch')
    expected_sources = {}
    source = CONFIG['sources']['bactrian']
    for lang in LANGUAGES:
        expected_sources[lang] = {'repo': source['repo'], 'revision': source['revision'], 'filename': source['files'][lang]}
    source = CONFIG['sources']['multijail']
    expected_sources['multijail'] = {k: source[k] for k in ('repo', 'revision', 'filename')}
    if set(manifest.get('source_files', {})) != set(expected_sources):
        raise ValueError('Invalid source file manifest')
    for name, expected in expected_sources.items():
        actual = manifest['source_files'][name]
        if any(actual.get(k) != v for k, v in expected.items()) or not re.fullmatch('[0-9a-f]{64}', actual.get('sha256', '')):
            raise ValueError('Invalid source pin/hash')
    if set(manifest.get('files', {})) != set(DATA_FILES):
        raise ValueError('Incorrect bundle file structure')
    rows = {}
    for name in DATA_FILES:
        blob = (path / name).read_bytes()
        if _sha(blob) != manifest['files'][name].get('sha256'):
            raise ValueError(f'File hash mismatch: {name}')
        rows[name] = [json.loads(line) for line in blob.decode('utf-8').splitlines()]
        if len(rows[name]) != manifest['files'][name].get('rows') or len(rows[name]) != manifest['counts'].get(name[:-6]):
            raise ValueError('Incorrect row counts')
    candidate_count = manifest.get('candidate_families')
    if type(candidate_count) is not int or candidate_count < 1:
        raise ValueError('Invalid candidate family count')
    _families(rows['candidates.jsonl'], LANGUAGES, candidate_count, 'bactrian')
    attack = _families(rows['attacks.jsonl'], ATTACK_LANGUAGES, CONFIG['selection']['attack_families'], 'multijail')
    alignment = _families(rows['alignment.jsonl'], LANGUAGES, CONFIG['selection']['alignment_families'], 'multijail')
    if set(attack) & set(alignment):
        raise ValueError('Attack/alignment families overlap')
    for name, families in (('attack', attack), ('alignment', alignment)):
        categories = Counter(tag for family in families.values() for tag in set(family[0]['harm_categories']))
        if dict(categories) != manifest['multijail_statistics'].get(f'{name}_category_family_counts'):
            raise ValueError('Category counts mismatch')
    review = rows['translation_review.jsonl']
    expected_review_ids = list(attack)[:CONFIG['selection']['translation_review_families']]
    if len(review) != len(expected_review_ids) * 2:
        raise ValueError('Incorrect translation review count')
    for row, (identity, lang) in zip(review, ((i, lang) for i in expected_review_ids for lang in ('ko', 'zh'))):
        matched = {r['lang']: r for r in attack[identity]}
        if (row.get('family_id') != identity or row.get('lang') != lang
                or row.get('id') != f'{identity}:{lang}:translation-review'
                or row.get('english_instruction') != matched['en']['instruction']
                or row.get('translated_instruction') != matched[lang]['instruction']
                or row.get('review_status') != 'pending' or row.get('human_label') is not None
                or row.get('reviewer') is not None
                or any(row.get(key) != matched[lang][key] for key in
                       ('source', 'source_dataset', 'source_revision', 'harm_categories'))):
            raise ValueError('Invalid matched translation review structure')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--candidate-families', type=int, default=1000)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    manifest = verify_bundle(args.output) if args.verify else prepare_sources(args.output, args.candidate_families)
    print(_json({'verified': True, 'counts': manifest['counts']}))
    return manifest


if __name__ == '__main__':
    main()
