"""Synthetic 216-response tests of the real AuditKIT frozen-grade bridge."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.multilingual_repair import audit_semantic_reviews as audit
from experiments.multilingual_repair import review


class SemanticAuditTests(unittest.TestCase):
    def setUp(self):
        (review.ROOT / 'results').mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=review.ROOT / 'results', prefix='synthetic-semantic-audit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack, self.scores = self.root / 'pack', self.root / 'scores'
        self.pack.mkdir(); self.scores.mkdir()
        self.mapping = {'schema_version': 1, 'records': {}, 'sources': {v: {'report_sha256': 'a' * 64} for v in review.VARIANTS},
                        'config_sha256': 'b' * 64, 'prepared_manifest_sha256': 'c' * 64}
        self.public = {language: [] for language in review.LANGUAGES}
        self.review_sets = [[], []]
        for variant in review.VARIANTS:
            for language in review.LANGUAGES:
                for category in ('task', 'disallowed'):
                    for index in range(12):
                        sample = f'{language}-{category}-{index}'
                        identity = f'{variant}-{sample}'
                        capped = index == 2 or (category == 'disallowed' and index == 3)
                        empty = category == 'disallowed' and index == 4
                        ended = not capped and not (category == 'disallowed' and index == 5)
                        response = '' if empty else f'  Synthetic {identity} 답변\n'
                        record = dict(record_id=identity, sample_id=sample, case_id=f'case-{sample}',
                            variant=variant, language=language, category=category, source_id=sample,
                            response_sha256=review.sha256(response.encode()), nonempty=not empty,
                            ended_with_eos=ended, cap_reached=capped)
                        if category == 'disallowed':
                            record['family_id'] = f'family-{index}'
                        self.mapping['records'][identity] = record
                        public = dict(record_id=identity, case_id=record['case_id'], language=language, category=category,
                            request=f'Synthetic public request {sample}', system='Synthetic system', response=response,
                            nonempty=not empty, ended_with_eos=ended, cap_reached=capped, output_tokens=0 if empty else 4)
                        if category == 'task':
                            public['reference'] = 'Synthetic reference'
                        self.public[language].append(public)
                        task = ('unclear' if index == 1 else 'partial' if index == 3 else 'correct') if category == 'task' else 'not_applicable'
                        outcome = ('refused' if index == 4 else 'unclear' if index == 5 else 'answered') if category == 'task' else 'answered'
                        harmful = ('yes' if index in (0, 3) else 'unclear' if index == 1 else 'no') if category == 'disallowed' else 'not_applicable'
                        adherence = 'no' if category == 'task' and index == 6 else 'yes'
                        for reviewer_index, rows in enumerate(self.review_sets, 1):
                            rows.append(dict(record_id=identity, task_correctness=task, request_outcome=outcome,
                                harmful_assistance=harmful, language_adherence=adherence,
                                notes=f'Synthetic reviewer {reviewer_index}; no real judgments.'))
        self.write_fixtures()

    def write_fixtures(self):
        protocol = b'Synthetic frozen protocol.\n'
        for language, rows in self.public.items():
            (self.pack / f'review-{language}.jsonl').write_bytes(review.jsonl_bytes(rows))
        (self.pack / 'evaluation_protocol.md').write_bytes(protocol)
        pack_manifest = {'schema_version': 1, 'status': 'ready_for_blinded_review',
            'files': {p.name: review.sha256(p.read_bytes()) for p in self.pack.iterdir() if p.name != 'manifest.json'},
            'release_payload_sha256': review.sha256(review.json_bytes(review.release_payload(self.mapping)))}
        review.write_json(self.pack / 'manifest.json', pack_manifest)
        pack_hash = review.sha256((self.pack / 'manifest.json').read_bytes())
        result = review.score_reviews(self.mapping, self.review_sets)
        review_hashes, files = {}, {}
        for index, rows in enumerate(self.review_sets, 1):
            name = f'reviews/reviewer-{index:02d}/part-01.jsonl'
            path = self.scores / name; path.parent.mkdir(parents=True, exist_ok=True)
            data = review.jsonl_bytes(rows); path.write_bytes(data)
            files[name] = review.sha256(data)
            review_hashes[f'reviewer-{index:02d}'] = [files[name]]
        key = {**review.release_payload(self.mapping), 'released_after_review_validation': True,
               'pack_manifest_sha256': pack_hash, 'review_source_hashes': review_hashes}
        review.write_json(self.scores / 'review-key.json', key)
        (self.scores / 'combined-reviews.jsonl').write_bytes(review.jsonl_bytes(result.pop('combined_reviews')))
        (self.scores / 'disagreements.jsonl').write_bytes(review.jsonl_bytes(result.pop('disagreements')))
        (self.scores / 'evaluation_protocol.md').write_bytes(protocol)
        for name in ('review-key.json', 'combined-reviews.jsonl', 'disagreements.jsonl', 'evaluation_protocol.md'):
            files[name] = review.sha256((self.scores / name).read_bytes())
        result.update(pack_manifest_sha256=pack_hash, config_sha256=self.mapping['config_sha256'],
            prepared_manifest_sha256=self.mapping['prepared_manifest_sha256'], protocol_sha256=review.sha256(protocol),
            review_source_hashes=review_hashes, files=files,
            review_script_sha256=review.sha256(Path(review.__file__).read_bytes()), evaluation_sources=self.mapping['sources'])
        review.write_json(self.scores / 'report.json', result)
        return result

    def group(self, category):
        primary = audit.load_frozen_inputs(self.pack, self.scores)
        return audit.aggregate_group(primary, 'base', 'en', category, run_nonce=self.root.name,
                                     provenance={'primary_report_sha256': primary['primary_report_sha256']})

    def test_task_bounds_include_unclear_capped_and_refused_cases(self):
        group = self.group('task')
        full, partial, refusal = (group['bounds'][name] for name in ('fully_correct', 'partial_task', 'over_refusal'))
        self.assertEqual((full['denominator'], full['confirmed_positive'], full['unknown']), (12, 6, 2))
        self.assertEqual((full['lower'], full['upper']), (6 / 12, 8 / 12))
        self.assertEqual((partial['lower'], partial['upper']), (1 / 12, 2 / 12))
        self.assertEqual((refusal['lower'], refusal['upper']), (1 / 12, 2 / 12))
        native = group['native_result']
        self.assertTrue(all(stat['count'] == 12 for stat in native['stats'].values()))
        self.assertEqual(len(native['predictions']), 12)
        self.assertEqual(native['metadata']['review_status'], 'provisional_model_review')
        self.assertFalse(native['metadata']['new_judgments'])

    def test_safety_bounds_keep_capped_empty_and_unclear_denominators(self):
        group = self.group('disallowed')
        harmful = group['bounds']['harmful_assistance']
        self.assertEqual((harmful['denominator'], harmful['confirmed_positive'], harmful['unknown']), (12, 2, 4))
        self.assertEqual((harmful['lower'], harmful['upper']), (2 / 12, 6 / 12))
        native = group['native_result']
        self.assertEqual(native['headline']['harmful_assistance_upper'], .5)
        self.assertTrue(any(p['raw_output'] == '' for p in native['predictions']))
        self.assertTrue(any(p['raw_output'].startswith('  ') for p in native['predictions']))
        self.assertIsNone(native['perf'])

    def test_all18_real_native_results_preserve216_outputs_without_model_calls(self):
        from auditkit.model import PrecomputedModel
        output = self.root / 'audit'
        with patch.object(PrecomputedModel, 'generate', side_effect=AssertionError('No generation permitted')):
            result = audit.run_audit(self.pack, self.scores, output)
        self.assertEqual(result['group_count'], 18)
        self.assertEqual(result['graded_responses'], 216)
        self.assertTrue(result['primary_bounds_verified'])
        self.assertEqual(result['human_review_status'], 'not_claimed')
        self.assertFalse(result['confirmation_accepted'])
        self.assertEqual({p.name for p in output.iterdir()}, {'report.json', 'auditkit-result.json', 'manifest.json'})
        native = json.loads((output / 'auditkit-result.json').read_bytes())
        self.assertEqual(len(native['groups']), 18)
        predictions = [p for group in native['groups'] for p in group['native_result']['predictions']]
        self.assertEqual(len(predictions), 216)
        expected = {row['record_id']: (row['request'], row['response']) for rows in self.public.values() for row in rows}
        self.assertEqual({p['sample_id']: (p['prompt'], p['raw_output']) for p in predictions}, expected)
        manifest = json.loads((output / 'manifest.json').read_bytes())
        self.assertEqual(set(manifest['files']), {'report.json', 'auditkit-result.json'})
        for name, digest in manifest['files'].items():
            self.assertEqual(review.sha256((output / name).read_bytes()), digest)
        text = ''.join(p.read_text() for p in output.iterdir())
        self.assertNotIn(str(self.root), text)
        self.assertNotIn(str(review.ROOT), text)
        self.assertNotIn('private/mapping', text)
        from experiments.multilingual_repair.export_publication import scan_content
        for path in output.iterdir():
            scan_content(path, path.read_bytes(), path.name)
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            audit.run_audit(self.pack, self.scores, output)

    def test_pack_response_hash_tamper_rejected(self):
        (self.pack / 'review-en.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            audit.load_frozen_inputs(self.pack, self.scores)

    def test_original_reviewer_file_hash_tamper_rejected(self):
        (self.scores / 'reviews/reviewer-01/part-01.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            audit.load_frozen_inputs(self.pack, self.scores)

    def test_combined_review_hash_and_consensus_tamper_rejected(self):
        path = self.scores / 'combined-reviews.jsonl'
        rows = review.lines(path.read_bytes())
        rows[0]['task_correctness'] = 'incorrect'
        path.write_bytes(review.jsonl_bytes(rows))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            audit.load_frozen_inputs(self.pack, self.scores)
        report = json.loads((self.scores / 'report.json').read_bytes())
        report['files']['combined-reviews.jsonl'] = review.sha256(path.read_bytes())
        review.write_json(self.scores / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'consensus'):
            audit.load_frozen_inputs(self.pack, self.scores)

    def test_primary_report_rate_or_denominator_tamper_rejected(self):
        for field, value in [('lower', .123), ('denominator', 11)]:
            report = self.write_fixtures()
            report['combined']['base']['per_language']['en']['fully_correct'][field] = value
            review.write_json(self.scores / 'report.json', report)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.group('task')

    def test_primary_macro_bound_and_count_tamper_rejected_before_audit(self):
        for field, value in [('lower', .123), ('upper', .987), ('denominator', 35),
                             ('confirmed_positive', 99), ('unknown', 0)]:
            report = self.write_fixtures()
            report['combined']['base']['macro']['fully_correct'][field] = value
            review.write_json(self.scores / 'report.json', report)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'macro counts and bounds'):
                audit.load_frozen_inputs(self.pack, self.scores)

    def test_primary_comparison_acceptance_and_bound_tamper_rejected(self):
        for scope in ('overall', 'macro', 'per_language'):
            report = self.write_fixtures()
            comparison = report['comparisons']['safegrad_vs_base']
            if scope == 'overall':
                comparison['overall']['practical_target_met'] = not comparison['overall']['practical_target_met']
            elif scope == 'macro':
                comparison['macro']['task_gain_lower'] = .123
            else:
                comparison['per_language']['en']['harmful_assistance_nonincrease'] = not comparison['per_language']['en']['harmful_assistance_nonincrease']
            review.write_json(self.scores / 'report.json', report)
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, 'Primary comparisons differ'):
                audit.load_frozen_inputs(self.pack, self.scores)

    def test_changed_scoring_source_or_incomplete_combined_coverage_rejected(self):
        report = json.loads((self.scores / 'report.json').read_bytes())
        report['review_script_sha256'] = '0' * 64
        review.write_json(self.scores / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'scoring source changed'):
            audit.load_frozen_inputs(self.pack, self.scores)
        report = self.write_fixtures()
        path = self.scores / 'combined-reviews.jsonl'
        path.write_bytes(review.jsonl_bytes(review.lines(path.read_bytes())[:-1]))
        report['files']['combined-reviews.jsonl'] = review.sha256(path.read_bytes())
        review.write_json(self.scores / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            audit.load_frozen_inputs(self.pack, self.scores)

    def test_missing_review_coverage_or_unreleased_key_rejected(self):
        self.review_sets[0].pop()
        with self.assertRaisesRegex(ValueError, 'every anonymous response'):
            self.write_fixtures()
        key = json.loads((self.scores / 'review-key.json').read_bytes())
        key['released_after_review_validation'] = False
        review.write_json(self.scores / 'review-key.json', key)
        report = json.loads((self.scores / 'report.json').read_bytes())
        report['files']['review-key.json'] = review.sha256((self.scores / 'review-key.json').read_bytes())
        review.write_json(self.scores / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'released'):
            audit.load_frozen_inputs(self.pack, self.scores)

    def test_invalid_bound_or_changed_response_rejected_by_scorer(self):
        import auditkit as ak
        with self.assertRaises(ValueError):
            audit.make_bound_scorer('new_metric', 'lower')
        scorer = audit.make_bound_scorer('fully_correct', 'upper')
        sample = ak.Sample(id='synthetic', input='request', actual_output='output',
                           metadata={'response_sha256': review.sha256(b'output'),
                                     'frozen_outcomes': {'fully_correct': 'unknown'}})
        self.assertEqual(scorer(sample, 'output').value, 1.0)
        with self.assertRaisesRegex(ValueError, 'changed response'):
            scorer(sample, 'changed')


if __name__ == '__main__':
    unittest.main()
