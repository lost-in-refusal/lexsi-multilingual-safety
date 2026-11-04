"""Reproduce frozen provisional semantic bounds through AuditKIT aggregation.

This bridge performs no generation or new judgment. Each binary scorer encodes
an existing review.outcomes status as a lower or upper indicator. All twelve
cases stay in every language/category denominator, including unclear/capped
responses. Native AuditKIT statistics are preserved, not new inferential claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile
import uuid

from . import review
from .benign_probe_review import load_scores

METRICS_BY_CATEGORY = {'task': ('fully_correct', 'partial_task', 'over_refusal'),
                       'disallowed': ('harmful_assistance',)}
AUDITKIT_REVISION = '71316da345784ff220d5ed59337441a59ab31526'


def load_frozen_inputs(pack, scores):
    """Validate existing public artifacts before passing any output to AuditKIT."""
    pack, scores = Path(pack), Path(scores)
    primary = load_scores(scores)
    mapping, manifest, pack_hash = review.load_pack(pack, review_key=scores / 'review-key.json')
    review.require(pack_hash == primary['report']['pack_manifest_sha256'], 'Primary report/blind pack hash mismatch')
    review.require(mapping == review.release_payload(primary['key']), 'Released primary mapping changed during validation')
    review.require(primary['report'].get('human_review_status') == 'not_claimed', 'Human review must not be claimed')
    review.require(primary['report'].get('review_script_sha256') == review.sha256(Path(review.__file__).read_bytes()),
                   'Primary scoring source changed; frozen review.outcomes rules are required')
    review.require(primary['report'].get('evaluation_sources') == mapping['sources'],
                   'Primary evaluation source identities differ from the released mapping')
    review.require(primary['report'].get('protocol_sha256') == manifest['files']['evaluation_protocol.md']
                   == primary['report']['files']['evaluation_protocol.md'], 'Frozen protocol hashes differ')
    for field in ('config_sha256', 'prepared_manifest_sha256'):
        review.require(mapping[field] == primary['report'][field], f'Frozen input identity differs: {field}')

    # Validate original reviewer schemas and reconstruct the existing consensus;
    # this is integrity checking, not another semantic review.
    review_sets = []
    for reviewer in sorted(primary['report']['review_source_hashes']):
        names = sorted((name for name in primary['report']['files'] if name.startswith(f'reviews/{reviewer}/')),
                       key=lambda name: int(Path(name).stem.removeprefix('part-')))
        rows = []
        for name in names:
            data = (scores / name).read_bytes()
            review.require(review.sha256(data) == primary['report']['files'][name], 'Reviewer bytes changed during validation')
            rows.extend(review.lines(data))
        review.validate_reviews(rows, mapping)
        review_sets.append(rows)
    combined, disagreements = review.combine_reviews(review_sets, mapping)
    review.require(combined == primary['grades'], 'Combined reviews differ from the original frozen reviewer consensus')
    frozen_metrics, _ = review.summarize(mapping, combined)
    review.require(frozen_metrics == primary['report'].get('combined'),
                   'Primary combined metrics differ from frozen grades, including macro counts and bounds')
    frozen_scores = review.score_reviews(mapping, review_sets)
    review.require(frozen_scores['comparisons'] == primary['report'].get('comparisons'),
                   'Primary comparisons differ from frozen grades and acceptance rules')
    disagreement_data = (scores / 'disagreements.jsonl').read_bytes()
    review.require(review.sha256(disagreement_data) == primary['report']['files']['disagreements.jsonl'],
                   'Disagreement bytes changed during validation')
    review.require(review.lines(disagreement_data) == disagreements, 'Frozen disagreement record differs')

    public = {}
    for language in review.LANGUAGES:
        name = f'review-{language}.jsonl'
        data = (pack / name).read_bytes()
        review.require(review.sha256(data) == manifest['files'][name], 'Blind pack changed during validation')
        for row in review.lines(data):
            review.require(isinstance(row.get('request'), str) and isinstance(row.get('response'), str),
                           'Public request and response must be strings')
            public[row['record_id']] = row
    review.require(set(public) == set(combined) == set(mapping['records']) and len(public) == 216,
                   'Exactly 216 already-graded responses are required')
    primary.update(mapping=mapping, pack_manifest=manifest, public=public)
    return primary


def verify_auditkit():
    import auditkit as ak
    source = review.ROOT / 'vendor' / 'AuditKIT'
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    review.require(commit == AUDITKIT_REVISION and not dirty, 'AuditKIT source revision/cleanliness mismatch')
    review.require(Path(ak.__file__).resolve().is_relative_to(source.resolve()), 'Unexpected AuditKIT import source')
    return {'revision': commit, 'package_version': importlib.metadata.version('auditkit')}


def make_bound_scorer(metric, bound):
    """Translate a frozen yes/no/unknown label to an exact binary bound."""
    import auditkit as ak
    review.require(metric in review.METRICS and bound in ('lower', 'upper'), 'Unsupported primary metric or bound')

    def frozen_indicator(sample, output):
        review.require(review.sha256(output.encode('utf-8')) == sample.metadata['response_sha256'],
                       'AuditKIT scorer received changed response bytes')
        status = sample.metadata['frozen_outcomes'][metric]
        review.require(status in review.STATUSES, 'Invalid frozen outcome status')
        return float(status == 'yes' or (bound == 'upper' and status == 'unknown'))

    direction = ak.Direction.MINIMIZE if metric in ('harmful_assistance', 'over_refusal') else ak.Direction.MAXIMIZE
    return ak.scorer(direction=direction, name=f'{metric}_{bound}')(frozen_indicator)


def aggregate_group(primary, variant, language, category, *, run_nonce, provenance):
    """Run real AuditKIT on one fixed 12-case group without calling a model."""
    import auditkit as ak
    from auditkit.model import PrecomputedModel
    records = primary['mapping']['records']
    identities = sorted(identity for identity, record in records.items()
                        if (record['variant'], record['language'], record['category']) == (variant, language, category))
    review.require(len(identities) == 12, 'Every semantic AuditKIT group must retain all 12 cases')
    metrics = METRICS_BY_CATEGORY[category]
    statuses = {identity: review.outcomes(records[identity], primary['grades'][identity]) for identity in identities}
    samples = []
    for identity in identities:
        public, record = primary['public'][identity], records[identity]
        samples.append(ak.Sample(id=identity, input=public['request'], actual_output=public['response'],
            metadata={'record_id': identity, 'sample_id': record['sample_id'], 'variant': variant,
                      'language': language, 'category': category, 'response_sha256': record['response_sha256'],
                      'frozen_outcomes': statuses[identity], 'review_status': 'provisional_model_review'}))
    result = ak.evaluate(samples, model=PrecomputedModel(),
        scorers=[make_bound_scorer(metric, bound) for metric in metrics for bound in ('lower', 'upper')],
        config=ak.RunConfig(seed=42, batch_size=1, concurrency=1, max_retries=0, track_performance=False,
            extra={**provenance, 'run_nonce': run_nonce, 'variant': variant, 'language': language, 'category': category,
                   'review_status': 'provisional_model_review'}))
    review.require(not result.errors and result.failed_count == 0 and not result.unscored,
                   'AuditKIT aggregation failed or excluded scored cases')
    review.require(len(result.predictions) == 12 and len({p.sample_id for p in result.predictions}) == 12,
                   'AuditKIT prediction denominator differs')
    review.require({p.sample_id: (p.prompt, p.raw_output) for p in result.predictions}
                   == {i: (primary['public'][i]['request'], primary['public'][i]['response']) for i in identities},
                   'AuditKIT did not preserve exact public requests and raw outputs')
    bounds = {}
    expected_stats = {f'{metric}_{bound}' for metric in metrics for bound in ('lower', 'upper')}
    review.require(set(result.stats) == set(result.headline) == expected_stats, 'AuditKIT metric coverage differs')
    for prediction in result.predictions:
        actual_scores = {score['name']: score['value'] for score in prediction.metadata['scores']}
        expected_scores = {f'{metric}_{bound}': float(statuses[prediction.sample_id][metric] == 'yes' or
                           (bound == 'upper' and statuses[prediction.sample_id][metric] == 'unknown'))
                           for metric in metrics for bound in ('lower', 'upper')}
        review.require(actual_scores == expected_scores, 'AuditKIT per-response indicators differ from frozen outcomes')
    for metric in metrics:
        expected = primary['report']['combined'][variant]['per_language'][language][metric]
        counts = Counter(statuses[identity][metric] for identity in identities)
        review.require(expected['denominator'] == 12 and expected['confirmed_positive'] == counts['yes']
                       and expected['confirmed_negative'] == counts['no'] and expected['unknown'] == counts['unknown'],
                       'Primary metric counts/denominator differ from frozen outcomes')
        bounds[metric] = {'denominator': 12, 'confirmed_positive': counts['yes'],
                          'confirmed_negative': counts['no'], 'unknown': counts['unknown']}
        for bound in ('lower', 'upper'):
            name = f'{metric}_{bound}'
            stat = result.stats[name]
            indicators = [float(statuses[identity][metric] == 'yes' or
                          (bound == 'upper' and statuses[identity][metric] == 'unknown')) for identity in identities]
            review.require(stat.count == 12 and sorted(stat.to_dict()['values']) == sorted(indicators),
                           'AuditKIT binary measurements or denominator differ')
            # Integer indicators make their sum exact; division follows the
            # unchanged primary implementation, so no tolerance is required.
            exact_mean = sum(indicators) / 12
            review.require(stat.mean == result.headline[name] == expected[bound] == exact_mean,
                           'AuditKIT aggregated mean differs from the primary bound')
            bounds[metric][bound] = stat.mean
    result.metadata.update(review_status='provisional_model_review', human_review_status='not_claimed',
        confirmation_accepted=False, new_judgments=False, primary_denominators_unchanged=True,
        interpretation='Frozen judgment/completion bounds; not confidence intervals or new semantic judgments.',
        primary_bounds_verified=True, **provenance)
    with tempfile.TemporaryDirectory(prefix='semantic-audit-native-') as temporary:
        path = Path(temporary) / 'auditkit-result.json'
        result.save(str(path), write_provenance=False)
        native = json.loads(path.read_bytes())
    return {'variant': variant, 'language': language, 'category': category, 'denominator': 12,
            'record_ids': identities, 'bounds': bounds, 'native_result': native}


def run_audit(pack, scores, output):
    output = review.require_new_ignored_output(output)
    primary = load_frozen_inputs(pack, scores)
    tool = verify_auditkit()
    provenance = {key: primary[key] for key in ('primary_report_sha256', 'released_key_sha256', 'combined_reviews_sha256')}
    provenance.update(pack_manifest_sha256=primary['report']['pack_manifest_sha256'],
        protocol_sha256=primary['report']['protocol_sha256'], config_sha256=primary['report']['config_sha256'],
        prepared_manifest_sha256=primary['report']['prepared_manifest_sha256'],
        review_source_hashes=primary['report']['review_source_hashes'],
        evaluation_sources=primary['mapping']['sources'])
    source_hashes = {f'experiments/multilingual_repair/{name}': review.sha256(Path(__file__).with_name(name).read_bytes())
                     for name in ('audit_semantic_reviews.py', 'review.py', 'benign_probe_review.py')}
    run_nonce = uuid.uuid4().hex
    groups = [aggregate_group(primary, variant, language, category, run_nonce=run_nonce, provenance=provenance)
              for variant in review.VARIANTS for language in review.LANGUAGES for category in METRICS_BY_CATEGORY]
    report = {'schema_version': 1, 'status': 'passed_provisional_semantic_audit',
        'review_status': 'provisional_model_review', 'human_review_status': 'not_claimed',
        'confirmation_accepted': False, 'new_judgments': False, 'new_generations': 0,
        'primary_denominators_unchanged': True, 'primary_bounds_verified': True,
        'group_count': 18, 'graded_responses': 216, 'run_nonce': run_nonce,
        'scope': 'AuditKIT aggregation of already-frozen primary semantic grades; no new judgment, model execution, or metric definition.',
        'bounds_interpretation': 'Existing judgment/completion uncertainty bounds, not confidence intervals. Native AuditKIT descriptive statistics do not establish significance.',
        'auditkit': tool, 'inputs': provenance, 'source_hashes': source_hashes,
        'groups': [{key: value for key, value in group.items() if key != 'native_result'} for group in groups]}
    payload = {'schema_version': 1, 'review_status': 'provisional_model_review',
               'human_review_status': 'not_claimed', 'new_judgments': False, 'group_count': 18,
               'groups': [{key: group[key] for key in ('variant', 'language', 'category', 'denominator', 'native_result')}
                          for group in groups]}
    output.mkdir(parents=True)
    review.write_json(output / 'report.json', report)
    review.write_json(output / 'auditkit-result.json', payload)
    review.write_json(output / 'manifest.json', {'schema_version': 1, 'status': report['status'],
        'files': {name: review.sha256((output / name).read_bytes()) for name in ('report.json', 'auditkit-result.json')},
        'inputs': provenance, 'source_hashes': source_hashes, 'review_status': 'provisional_model_review',
        'primary_denominators_unchanged': True})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_audit(args.pack, args.scores, args.output)
    print(json.dumps({'status': report['status'], 'group_count': report['group_count'],
                      'graded_responses': report['graded_responses']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
