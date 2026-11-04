#!/usr/bin/env python3
"""Copy an explicit scientific package to a new ignored directory; never publish it.

Original bytes and hashes are preserved. Unsafe artifacts are rejected rather
than redacted. Confirmation texts, binary weights, legacy evidence, Git history,
and unlisted files are never copied. Python source is exempt from generic marker
checks so source and tests can define rejection rules. Private deny terms still
apply to every selected file, including Python.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


PACKAGE = Path('experiments/multilingual_repair')
REQUIRED_SOURCE = ('prepare.py', 'train.py', 'evaluate.py', 'export_publication.py',
                   'config.json', 'requirements.txt', 'README.md', 'evaluation_protocol.md',
                   'DATA_ATTRIBUTION.md', 'licenses/Apache-2.0.txt',
                   'licenses/MultiJail-MIT.txt', 'licenses/Aya-dataset-card.md')
PREPARED_FILES = ('config.json', 'manifest.json', 'train.jsonl', 'calibration.jsonl',
                  'development.jsonl', 'alignment.jsonl')
RUN_FILES = ('report.json', 'steps.jsonl', 'responses.jsonl', 'auditkit-result.json',
             'training-config.json', 'prepared-config.json', 'prepared-manifest.json',
             'training-source.py', 'evaluation-source.py', 'summary.json', 'annotations.json', 'inputs.jsonl',
             'eligibility.json', 'manifest.json')
REVIEW_PACK_FILES = ('review-en.jsonl', 'review-ko.jsonl', 'review-zh.jsonl', 'evaluation_protocol.md')
REVIEW_SCORE_FILES = frozenset(('review-key.json', 'combined-reviews.jsonl',
                                'disagreements.jsonl', 'evaluation_protocol.md'))
REVIEW_PART = re.compile(r'reviews/reviewer-[0-9]{2}/part-[0-9]{2}\.jsonl\Z')
DENIED_FIELDS = frozenset(('answer_verbatim', 'question_presented', 'question_item_id',
                          'conversation_id', 'chat_id', 'thread_id', 'chat_transcript',
                          'user_reply', 'human_reviewer', 'source_thread'))
DENIED_MARKERS = re.compile(
    r'\bcall_[A-Za-z0-9_-]+|request_user_input|private-planning', re.I)
ABSOLUTE_PATH = re.compile(
    r'(?<![A-Za-z0-9:/])/(?:Users|home|tmp|private|var|opt|usr|mnt|root)(?:/|\b)|'
    r'(?<![A-Za-z0-9:/])/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]*|'
    r'\b[A-Za-z]:[\\/]|(?<!\w)file://', re.I)
SECRET_TOKEN = re.compile(
    r'(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|'
    r'github_pat_[A-Za-z0-9_]{40,}|sk[_-](?:proj[-_])?[A-Za-z0-9_-]{24,})')
TEXT_FIELDS = frozenset(('request', 'prompt', 'reference', 'response', 'responses',
                         'messages', 'text', 'content', 'input_ids', 'labels'))
ALIAS = re.compile(r'[a-z0-9][a-z0-9_-]{0,63}\Z')
DIGEST = re.compile(r'[0-9a-f]{64}\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def safe_path(root, path):
    """Reject symlinks at every project-relative component before resolving."""
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError('Path must be inside the project') from None
    require('..' not in relative.parts, 'Path traversal is forbidden')
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), 'A selected path contains a symlink')
    require(path.resolve().is_relative_to(root.resolve()), 'Path escapes project')
    return path


def scan_string(value, label, deny_terms=(), *, check_paths=True):
    require(not SECRET_TOKEN.search(value), f'{label}: plausible credential token')
    require(not DENIED_MARKERS.search(value), f'{label}: private or dialogue marker')
    if check_paths:
        require(not ABSOLUTE_PATH.search(value), f'{label}: absolute filesystem path')
    require(not any(term.casefold() in value.casefold() for term in deny_terms),
            f'{label}: caller-supplied private term')


def scan_object(value, label, confirmation=False, deny_terms=()):
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        require(not keys.intersection(DENIED_FIELDS), f'{label}: forbidden field')
        confirmation = confirmation or any(
            str(value.get(key, '')).lower() in ('confirmation', 'test', 'heldout', 'held_out')
            for key in ('split', 'partition', 'phase'))
        require(not (confirmation and keys.intersection(TEXT_FIELDS)), f'{label}: confirmation text')
        for key, child in value.items():
            scan_string(str(key), label, deny_terms)
            scan_object(child, label, confirmation or 'confirmation' in str(key).lower(), deny_terms)
    elif isinstance(value, list):
        for child in value:
            scan_object(child, label, confirmation, deny_terms)
    elif isinstance(value, str):
        scan_string(value, label, deny_terms)
        require(not value.startswith(('/', '\\\\')), f'{label}: absolute filesystem path')


def scan_content(path, blob, label, deny_terms=()):
    try:
        text = blob.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError(f'{label}: non-UTF-8 artifact') from None
    require(not SECRET_TOKEN.search(text), f'{label}: plausible credential token')
    if path.suffix == '.py':
        require(not any(term.casefold() in text.casefold() for term in deny_terms),
                f'{label}: caller-supplied private term')
        return
    is_json = path.suffix in ('.json', '.jsonl')
    # JSON escapes are syntax; only decoded keys and values represent paths.
    # Other raw privacy checks still apply before recursively scanning values.
    scan_string(text, label, deny_terms, check_paths=not is_json)
    if is_json:
        def invalid_constant(_):
            raise ValueError('Non-finite JSON number')

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate JSON key')
                result[key] = value
            return result

        try:
            values = [json.loads(line, parse_constant=invalid_constant, object_pairs_hook=unique_object)
                      for line in text.splitlines() if line.strip()] if path.suffix == '.jsonl' else [
                          json.loads(text, parse_constant=invalid_constant, object_pairs_hook=unique_object)]
        except ValueError:
            raise ValueError(f'{label}: invalid JSON') from None
        for value in values:
            scan_object(value, label, deny_terms=deny_terms)


def export_publication(project_root, output, *, prepared=None, runs=None, review_packs=None,
                       review_scores=None, deny_terms=()):
    """Export exact allowlisted files, returning the new publication manifest.

    prepared/runs map public aliases to explicit directories under results/.
    No directory traversal, Git operation, artifact rewriting, or model load is
    performed. All selected bytes are validated before output creation.
    """
    # Keep the caller's spelling so macOS /var -> /private/var aliases work;
    # containment is separately checked against resolved paths in safe_path.
    root = Path(project_root).absolute()
    require(isinstance(deny_terms, (list, tuple))
            and all(isinstance(term, str) and term.strip() for term in deny_terms),
            'Caller-supplied deny terms must be nonempty strings')
    output = safe_path(root, output)
    require(not output.exists(), 'Output already exists; use a new directory')
    require(output.is_relative_to(root / 'results') and output != root / 'results',
            'Output must be inside results/')
    ignored = subprocess.run(['git', '-C', str(root), 'check-ignore', '--quiet', '--',
                              output.relative_to(root).as_posix()], capture_output=True)
    require(ignored.returncode == 0, 'Output must be ignored by Git')
    selected, withheld = {}, []

    def add(source, target):
        source = safe_path(root, source)
        require(source.is_file(), f'{target}: required file missing')
        require(target not in selected, f'{target}: duplicate export destination')
        scan_string(source.relative_to(root).as_posix(), target, deny_terms)
        blob = source.read_bytes()
        scan_content(source, blob, target, deny_terms)
        selected[target] = (blob, source.relative_to(root).as_posix())

    package = safe_path(root, PACKAGE)
    for name in REQUIRED_SOURCE:
        add(package / name, (PACKAGE / name).as_posix())
    add(package / 'README.md', 'README.md')
    add(root / '.gitignore', '.gitignore')
    for path in sorted(package.glob('*.py')):
        target = (PACKAGE / path.name).as_posix()
        if target not in selected:
            add(path, target)
    tests = sorted((root / 'tests').glob('test_multilingual*.py'))
    require(bool(tests), 'No multilingual experiment tests found')
    for path in tests:
        add(path, f'tests/{path.name}')

    def artifact_directory(alias, directory):
        require(isinstance(alias, str) and ALIAS.fullmatch(alias), 'Invalid public artifact alias')
        scan_string(alias, 'Artifact alias', deny_terms)
        directory = safe_path(root, directory)
        require(directory.is_dir() and directory.is_relative_to(root / 'results'),
                'Scientific artifact directories must be inside results/')
        require(not output.is_relative_to(directory), 'Output cannot be inside a selected artifact')
        scan_string(directory.relative_to(root).as_posix(), 'Selected artifact directory', deny_terms)
        require('evidence' not in directory.relative_to(root).parts
                and not any('adjudication' in part.lower() for part in directory.parts),
                'Legacy evidence and adjudication directories cannot be exported')
        return directory

    for kind, mapping, allowed in (('data', prepared or {}, PREPARED_FILES),
                                   ('runs', runs or {}, RUN_FILES)):
        for alias, directory in sorted(mapping.items()):
            directory = artifact_directory(alias, directory)
            base = f'{kind}/{alias}'
            required = PREPARED_FILES if kind == 'data' else ('report.json',)
            for name in allowed:
                path = safe_path(root, directory / name)
                if name in required or path.exists():
                    add(path, f'{base}/{name}')
            if kind == 'data':
                manifest = json.loads(selected[f'{base}/manifest.json'][0])
                require(manifest.get('status') == 'passed_tokenization_preflight',
                        f'{base}: preparation has not passed')
                files = manifest.get('files', {})
                require(isinstance(files, dict), f'{base}: invalid prepared file hashes')
                for name in PREPARED_FILES:
                    if name != 'manifest.json':
                        require(files.get(name) == digest(selected[f'{base}/{name}'][0]),
                                f'{base}/{name}: prepared hash mismatch')
                require(isinstance(files.get('confirmation.jsonl'), str)
                        and DIGEST.fullmatch(files['confirmation.jsonl']),
                        f'{base}: missing confirmation hash')
                withheld.append(f'{base}/confirmation.jsonl')
            else:
                report = json.loads(selected[f'{base}/report.json'][0])
                require(isinstance(report, dict), f'{base}: run report must be an object')
                require(not any('confirmation' in str(report.get(key, '')).lower()
                                or str(report.get(key, '')).lower() in ('test', 'heldout', 'held_out')
                                for key in ('split', 'partition', 'phase'))
                        and report.get('confirmation_generated_samples', 0) == 0,
                        f'{base}: confirmation run cannot be exported')
                pairs = [('responses_sha256', 'responses.jsonl'), ('steps_sha256', 'steps.jsonl'),
                         ('annotations_sha256', 'annotations.json'), ('inputs_sha256', 'inputs.jsonl'),
                         ('eligibility_sha256', 'eligibility.json'),
                         ('training_script_sha256', 'training-source.py')]
                pairs += [('config_sha256', name) for name in ('training-config.json', 'prepared-config.json')
                          if f'{base}/{name}' in selected]
                if f'{base}/prepared-manifest.json' in selected:
                    pairs.append(('prepared_manifest_sha256', 'prepared-manifest.json'))
                for field, name in pairs:
                    if field in report:
                        target = f'{base}/{name}'
                        require(target in selected, f'{target}: declared run artifact missing')
                        require(report[field] == digest(selected[target][0]),
                                f'{target}: declared run hash mismatch')
                if f'{base}/manifest.json' in selected:
                    run_manifest = json.loads(selected[f'{base}/manifest.json'][0])
                    declared = run_manifest.get('files')
                    require(isinstance(declared, dict) and bool(declared), f'{base}: invalid run manifest')
                    for name, expected in declared.items():
                        target = f'{base}/{name}'
                        require(name in RUN_FILES and name != 'manifest.json' and target in selected,
                                f'{base}: unapproved or missing run-manifest artifact')
                        require(expected == digest(selected[target][0]), f'{target}: run-manifest hash mismatch')

    packs_by_hash = {}
    for alias, directory in sorted((review_packs or {}).items()):
        directory = artifact_directory(alias, directory)
        base = f'review-packs/{alias}'
        add(directory / 'manifest.json', f'{base}/manifest.json')
        pack = json.loads(selected[f'{base}/manifest.json'][0])
        require(pack.get('status') == 'ready_for_blinded_review'
                and set(pack.get('files', {})) == set(REVIEW_PACK_FILES), f'{base}: invalid review pack manifest')
        for name in REVIEW_PACK_FILES:
            add(directory / name, f'{base}/{name}')
            require(pack['files'][name] == digest(selected[f'{base}/{name}'][0]),
                    f'{base}/{name}: review pack hash mismatch')
        packs_by_hash[digest(selected[f'{base}/manifest.json'][0])] = pack

    for alias, directory in sorted((review_scores or {}).items()):
        directory = artifact_directory(alias, directory)
        base = f'review-scores/{alias}'
        add(directory / 'report.json', f'{base}/report.json')
        report = json.loads(selected[f'{base}/report.json'][0])
        pack_hash = report.get('pack_manifest_sha256')
        require(pack_hash in packs_by_hash, f'{base}: export the matching review pack')
        files = report.get('files', {})
        require(report.get('status') == 'passed_provisional_review' and isinstance(files, dict)
                and REVIEW_SCORE_FILES <= set(files)
                and all(name in REVIEW_SCORE_FILES or REVIEW_PART.fullmatch(name) for name in files),
                f'{base}: invalid review score file manifest')
        for name, expected in sorted(files.items()):
            add(directory / name, f'{base}/{name}')
            require(digest(selected[f'{base}/{name}'][0]) == expected, f'{base}/{name}: review score hash mismatch')
        reviewer_hashes = {}
        for name in sorted(files):
            if REVIEW_PART.fullmatch(name):
                reviewer_hashes.setdefault(Path(name).parts[1], []).append(files[name])
        require(len(reviewer_hashes) >= 2 and reviewer_hashes == report.get('review_source_hashes'),
                f'{base}: independent review source hashes differ')
        key = json.loads(selected[f'{base}/review-key.json'][0])
        require(key.get('released_after_review_validation') is True and key.get('pack_manifest_sha256') == pack_hash
                and key.get('review_source_hashes') == reviewer_hashes, f'{base}: review key has not been validated')
        fields = ('schema_version', 'records', 'sources', 'config_sha256', 'prepared_manifest_sha256')
        require(all(field in key for field in fields), f'{base}: released review payload is incomplete')
        payload = {field: key[field] for field in fields}
        blob = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        require(digest(blob) == packs_by_hash[pack_hash].get('release_payload_sha256'),
                f'{base}: released review payload hash mismatch')
        require(files['evaluation_protocol.md'] == packs_by_hash[pack_hash]['files']['evaluation_protocol.md'],
                f'{base}: review protocol hash mismatch')

    manifest = {
        'schema_version': 1,
        'scope': 'Exact allowlisted scientific files; originals unchanged; no Git history included.',
        'content_scan': 'JSON, JSONL, and documentation scanned; Python exempt from generic markers; caller deny terms checked in every file.',
        'caller_supplied_deny_terms_checked': bool(deny_terms),
        'confirmation_texts_exported': False,
        'withheld_confirmation_files': withheld,
        'unlisted_files_and_binary_weights_exported': False,
        'exporter_sha256': digest(Path(__file__).read_bytes()),
        'files': {target: {'sha256': digest(blob), 'bytes': len(blob), 'source_relative_path': source}
                  for target, (blob, source) in sorted(selected.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        for target, (blob, _) in selected.items():
            destination = output / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(blob)
            require(digest(destination.read_bytes()) == digest(blob), f'{target}: copied hash mismatch')
        with (output / 'publication-manifest.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest


def named_directories(values):
    result = {}
    for value in values:
        require('=' in value, 'Artifact arguments must use alias=directory')
        alias, directory = value.split('=', 1)
        require(alias not in result, 'Duplicate artifact alias')
        require(bool(directory), 'Artifact directory is empty')
        result[alias] = Path(directory)
    return result


def load_deny_terms(project_root, path):
    """Read an explicitly supplied ignored local JSON list; never export it."""
    if path is None:
        return ()
    root = Path(project_root).absolute()
    path = safe_path(root, path)
    require(path.is_relative_to(root / 'results'), 'Deny-terms file must be under ignored results/')
    ignored = subprocess.run(['git', '-C', str(root), 'check-ignore', '--quiet', '--',
                              path.relative_to(root).as_posix()], capture_output=True)
    require(ignored.returncode == 0, 'Deny-terms file must be ignored by Git')
    try:
        terms = json.loads(path.read_text(encoding='utf-8'))
    except ValueError:
        raise ValueError('Deny-terms file must contain a JSON list') from None
    require(isinstance(terms, list) and all(isinstance(term, str) and term.strip() for term in terms),
            'Deny-terms file must contain only nonempty strings')
    return terms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared', action='append', default=[], metavar='ALIAS=DIRECTORY')
    parser.add_argument('--run', action='append', default=[], metavar='ALIAS=DIRECTORY')
    parser.add_argument('--review-pack', action='append', default=[], metavar='ALIAS=DIRECTORY')
    parser.add_argument('--review-scores', action='append', default=[], metavar='ALIAS=DIRECTORY')
    parser.add_argument('--deny-terms-file', type=Path, help='Ignored local JSON string list; never copied')
    args = parser.parse_args(argv)
    try:
        result = export_publication(args.project_root, args.output,
                                    prepared=named_directories(args.prepared), runs=named_directories(args.run),
                                    review_packs=named_directories(args.review_pack),
                                    review_scores=named_directories(args.review_scores),
                                    deny_terms=load_deny_terms(args.project_root, args.deny_terms_file))
    except (ValueError, OSError) as exc:
        # ValueErrors contain controlled relative identifiers, never the rejected data.
        parser.exit(1, f'Export rejected: {exc if isinstance(exc, ValueError) else type(exc).__name__}\n')
    print(json.dumps({'status': 'exported', 'files': len(result['files']),
                      'confirmation_texts_exported': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
