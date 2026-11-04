import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / 'experiments/multilingual_repair/export_publication.py'


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), 'Publication exporter entry point is missing')
        spec = importlib.util.spec_from_file_location('multilingual_publication', MODULE)
        self.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.exporter)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root / '.gitignore').write_text('results/\n')
        self.package = self.root / 'experiments/multilingual_repair'
        self.package.mkdir(parents=True)
        for name in ('prepare.py', 'train.py', 'evaluate.py', 'export_publication.py'):
            (self.package / name).write_text('# example source\n')
        for name in ('README.md', 'evaluation_protocol.md', 'requirements.txt', 'DATA_ATTRIBUTION.md',
                     'licenses/Apache-2.0.txt', 'licenses/MultiJail-MIT.txt', 'licenses/Aya-dataset-card.md'):
            path = self.package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Scientific experiment documentation.\n')
        (self.package / 'config.json').write_text('{"seed": 42}\n')
        (self.root / 'tests').mkdir()
        (self.root / 'tests/test_multilingual_example.py').write_text('assert True\n')
        self.output = self.root / 'results/publication'

    def prepare(self):
        directory = self.root / 'results/prepared'
        directory.mkdir(parents=True)
        (directory / 'config.json').write_bytes((self.package / 'config.json').read_bytes())
        for split in ('train', 'calibration', 'development', 'alignment'):
            (directory / f'{split}.jsonl').write_text(json.dumps({'split': split, 'request': 'An example task'}) + '\n')
        # This marker must never be read or copied by the exporter.
        (directory / 'confirmation.jsonl').write_text('{"answer_verbatim": "private"}\n')
        manifest = {'status': 'passed_tokenization_preflight',
                    'files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in directory.iterdir()}}
        (directory / 'manifest.json').write_text(json.dumps(manifest) + '\n')
        return directory

    def run_directory(self, report=None):
        directory = self.root / 'results/scientific-run'
        directory.mkdir(parents=True)
        (directory / 'report.json').write_text(json.dumps(report or {'status': 'passed_execution'}) + '\n')
        return directory

    def export(self, **kwargs):
        kwargs.setdefault('deny_terms', ('ExamplePerson', 'ExampleAccount'))
        return self.exporter.export_publication(self.root, self.output, **kwargs)

    def test_exact_allowlist_copy_and_manifest_hashes(self):
        (self.root / 'README.md').write_text('Private historical discussion')
        (self.package / 'private-notes.txt').write_text('Never publish this')
        (self.root / 'tests/test_legacy.py').write_text('Never publish this')
        original = (self.package / 'prepare.py').read_bytes()
        manifest = self.export()
        path = 'experiments/multilingual_repair/prepare.py'
        self.assertEqual((self.output / path).read_bytes(), original)
        self.assertEqual(manifest['files'][path]['sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(manifest['files'][path]['bytes'], len(original))
        self.assertEqual(manifest, json.loads((self.output / 'publication-manifest.json').read_text()))
        self.assertEqual((self.output / 'README.md').read_bytes(), (self.package / 'README.md').read_bytes())
        self.assertEqual((self.output / '.gitignore').read_bytes(), (self.root / '.gitignore').read_bytes())
        self.assertFalse((self.output / 'experiments/multilingual_repair/private-notes.txt').exists())
        self.assertFalse((self.output / 'tests/test_legacy.py').exists())
        self.assertIn('experiments/multilingual_repair/licenses/MultiJail-MIT.txt', manifest['files'])
        self.assertIn('experiments/multilingual_repair/DATA_ATTRIBUTION.md', manifest['files'])

    def test_source_can_define_denial_rules_without_exporting_private_data(self):
        (self.package / 'export_publication.py').write_text("DENIED = ['answer_verbatim', '/Users/example', 'call_example']\n")
        self.export()
        self.assertEqual((self.output / 'experiments/multilingual_repair/export_publication.py').read_bytes(),
                         (self.package / 'export_publication.py').read_bytes())

    def test_prepared_confirmation_withheld_and_originals_unchanged(self):
        prepared = self.prepare()
        original = {p.name: p.read_bytes() for p in prepared.iterdir()}
        manifest = self.export(prepared={'corpus': prepared})
        self.assertFalse((self.output / 'data/corpus/confirmation.jsonl').exists())
        self.assertEqual((self.output / 'data/corpus/manifest.json').read_bytes(), original['manifest.json'])
        self.assertIn('data/corpus/confirmation.jsonl', manifest['withheld_confirmation_files'])
        self.assertEqual(original, {p.name: p.read_bytes() for p in prepared.iterdir()})

    def test_nested_dialogue_fields_rejected_before_output_creation(self):
        for field in ('answer_verbatim', 'question_presented', 'question_item_id', 'thread_id'):
            with self.subTest(field=field):
                (self.package / 'config.json').write_text(json.dumps({'nested': [{field: 'private'}]}))
                with self.assertRaisesRegex(ValueError, 'forbidden field'):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_private_strings_and_absolute_paths_rejected(self):
        for content in ('call_example123456', '/Users/example/file', '/tmp/raw.log', '/scratch/model/checkpoint',
                        'C:\\Users\\person\\data', 'private-planning', 'ExamplePerson'):
            with self.subTest(content=content):
                (self.package / 'README.md').write_text(content)
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_encoded_json_private_string_rejected(self):
        (self.package / 'config.json').write_text('{"path": "\\u002fUsers\\u002fexample\\u002fdata"}')
        with self.assertRaisesRegex(ValueError, 'absolute filesystem path'):
            self.export()

    def test_json_colon_then_newline_is_exported_without_changing_bytes(self):
        run = self.run_directory({'status': 'passed_execution', 'diagnostic': 'y:\nSynthetic value'})
        original = (run / 'report.json').read_bytes()
        self.export(runs={'example': run})
        self.assertEqual((self.output / 'runs/example/report.json').read_bytes(), original)
        self.assertEqual((run / 'report.json').read_bytes(), original)

    def test_jsonl_colon_then_newline_is_exported_without_changing_bytes(self):
        run = self.run_directory()
        original = (json.dumps({'response': 'y:\nSynthetic value'}) + '\n').encode()
        (run / 'responses.jsonl').write_bytes(original)
        self.export(runs={'example': run})
        self.assertEqual((self.output / 'runs/example/responses.jsonl').read_bytes(), original)
        self.assertEqual((run / 'responses.jsonl').read_bytes(), original)

    def test_encoded_true_paths_in_json_keys_and_values_are_rejected(self):
        paths = ('/Users/example/data', '/scratch/model/checkpoint', r'C:\Users\example\data')
        for suffix in ('.json', '.jsonl'):
            for value in paths:
                encoded = ''.join('\\u%04x' % ord(char) for char in value)
                for field in ('key', 'value'):
                    content = ('{"' + encoded + '": "example"}' if field == 'key'
                               else '{"nested": [{"value": "' + encoded + '"}]}')
                    with self.subTest(suffix=suffix, path=value, field=field):
                        with self.assertRaisesRegex(ValueError, 'absolute filesystem path'):
                            self.exporter.scan_content(Path('artifact' + suffix), content.encode(), 'artifact')

    def test_encoded_json_privacy_markers_fields_and_tokens_are_rejected(self):
        cases = (('call_example123', 'private or dialogue marker'),
                 ('private-planning', 'private or dialogue marker'),
                 ('ExamplePerson', 'caller-supplied private term'),
                 ('hf_' + 'x' * 48, 'plausible credential token'))
        for suffix in ('.json', '.jsonl'):
            for value, reason in cases:
                encoded = ''.join('\\u%04x' % ord(char) for char in value)
                content = '{"nested": [{"value": "' + encoded + '"}]}'
                with self.subTest(suffix=suffix, reason=reason):
                    with self.assertRaisesRegex(ValueError, reason):
                        self.exporter.scan_content(Path('artifact' + suffix), content.encode(), 'artifact', ('ExamplePerson',))
            field = ''.join('\\u%04x' % ord(char) for char in 'answer_verbatim')
            with self.subTest(suffix=suffix, reason='forbidden field'):
                with self.assertRaisesRegex(ValueError, 'forbidden field'):
                    self.exporter.scan_content(Path('artifact' + suffix), ('{"' + field + '": "example"}').encode(), 'artifact')

    def test_duplicate_json_keys_cannot_hide_earlier_values(self):
        content = b'{"value": "\\u002fUsers\\u002fexample\\u002fdata", "value": "example"}'
        for suffix in ('.json', '.jsonl'):
            with self.subTest(suffix=suffix):
                with self.assertRaisesRegex(ValueError, 'invalid JSON'):
                    self.exporter.scan_content(Path('artifact' + suffix), content, 'artifact')

    def test_malformed_and_nonfinite_json_and_jsonl_are_rejected(self):
        for suffix in ('.json', '.jsonl'):
            for content in (b'{broken', b'{"value": NaN}', b'{"value": Infinity}', b'{"value": -Infinity}'):
                with self.subTest(suffix=suffix, content=content):
                    with self.assertRaisesRegex(ValueError, 'invalid JSON'):
                        self.exporter.scan_content(Path('artifact' + suffix), content, 'artifact')

    def test_confirmation_record_disguised_as_development_rejected(self):
        prepared = self.prepare()
        (prepared / 'development.jsonl').write_text('{"split":"confirmation", "request":"reserved"}\n')
        manifest = json.loads((prepared / 'manifest.json').read_text())
        manifest['files']['development.jsonl'] = hashlib.sha256((prepared / 'development.jsonl').read_bytes()).hexdigest()
        (prepared / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'confirmation text'):
            self.export(prepared={'corpus': prepared})
        self.assertFalse(self.output.exists())

    def test_tampered_prepared_file_rejected(self):
        prepared = self.prepare()
        (prepared / 'train.jsonl').write_text('{"request":"changed"}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.export(prepared={'corpus': prepared})

    def test_run_report_private_path_rejected_without_redaction(self):
        run = self.run_directory({'status': 'failed', 'error': 'Failed at /Users/example/private/file'})
        original = (run / 'report.json').read_bytes()
        with self.assertRaises(ValueError):
            self.export(runs={'plain': run})
        self.assertEqual((run / 'report.json').read_bytes(), original)
        self.assertFalse(self.output.exists())

    def test_only_known_run_files_are_copied(self):
        run = self.run_directory()
        (run / 'steps.jsonl').write_text('{"step":1,"loss":0.5}\n')
        (run / 'error.log').write_text('/Users/example/private')
        (run / 'adapter_model.safetensors').write_bytes(b'not a model')
        manifest = self.export(runs={'plain': run})
        self.assertIn('runs/plain/steps.jsonl', manifest['files'])
        self.assertFalse((self.output / 'runs/plain/error.log').exists())
        self.assertFalse((self.output / 'runs/plain/adapter_model.safetensors').exists())
        self.assertNotIn('error.log', json.dumps(manifest))

    def test_confirmation_run_cannot_leak_unlabeled_responses(self):
        run = self.run_directory({'status': 'passed_execution', 'split': 'confirmation'})
        (run / 'responses.jsonl').write_text('{"response":"reserved output"}\n')
        with self.assertRaisesRegex(ValueError, 'confirmation run'):
            self.export(runs={'reserved': run})
        self.assertFalse(self.output.exists())

    def test_run_artifact_hash_mismatches_rejected(self):
        run = self.run_directory({'status': 'passed_execution', 'responses_sha256': '0' * 64})
        (run / 'responses.jsonl').write_text('{"response":"changed"}\n')
        with self.assertRaisesRegex(ValueError, 'declared run hash mismatch'):
            self.export(runs={'plain': run})
        self.assertFalse(self.output.exists())

    def test_run_declared_steps_must_exist(self):
        run = self.run_directory({'status': 'passed_training', 'steps_sha256': 'a' * 64})
        with self.assertRaisesRegex(ValueError, 'declared run artifact missing'):
            self.export(runs={'plain': run})

    def test_secondary_eligibility_and_manifest_are_verified(self):
        eligibility = b'{"records": []}\n'
        run = self.run_directory({'status': 'provisional_diagnostic',
                                  'eligibility_sha256': hashlib.sha256(eligibility).hexdigest()})
        (run / 'eligibility.json').write_bytes(eligibility)
        manifest = {'files': {'eligibility.json': '0' * 64}}
        (run / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'run-manifest hash mismatch'):
            self.export(runs={'secondary': run})
        manifest['files']['eligibility.json'] = hashlib.sha256(eligibility).hexdigest()
        (run / 'manifest.json').write_text(json.dumps(manifest))
        result = self.export(runs={'secondary': run})
        self.assertIn('runs/secondary/eligibility.json', result['files'])

    def test_run_manifest_cannot_add_unapproved_artifacts(self):
        run = self.run_directory()
        (run / 'manifest.json').write_text(json.dumps({'files': {'confirmation.jsonl': '0' * 64}}))
        with self.assertRaisesRegex(ValueError, 'unapproved or missing run-manifest artifact'):
            self.export(runs={'secondary': run})

    def test_input_audit_artifacts_are_bound_to_the_report(self):
        annotation = b'{"records": []}\n'
        inputs = b'{"id": "example", "request": "Explain addition."}\n'
        report = {'status': 'provisional_model_input_audit', 'annotations_sha256': '0' * 64,
                  'inputs_sha256': '0' * 64}
        run = self.run_directory(report)
        (run / 'annotations.json').write_bytes(annotation)
        (run / 'inputs.jsonl').write_bytes(inputs)
        with self.assertRaisesRegex(ValueError, 'declared run hash mismatch'):
            self.export(runs={'input-audit': run})
        report['annotations_sha256'] = hashlib.sha256(annotation).hexdigest()
        (run / 'report.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'declared run hash mismatch'):
            self.export(runs={'input-audit': run})
        report['inputs_sha256'] = hashlib.sha256(inputs).hexdigest()
        (run / 'report.json').write_text(json.dumps(report))
        manifest = self.export(runs={'input-audit': run})
        self.assertEqual(manifest['files']['runs/input-audit/inputs.jsonl']['sha256'], report['inputs_sha256'])

    def test_run_config_and_training_source_hashes_must_match(self):
        source = b'# Frozen original source\n'
        run = self.run_directory({'status': 'passed_training', 'config_sha256': 'a' * 64,
                                  'training_script_sha256': hashlib.sha256(source).hexdigest()})
        (run / 'training-source.py').write_bytes(source)
        (run / 'training-config.json').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'declared run hash mismatch'):
            self.export(runs={'plain': run})

    def test_symlinks_rejected(self):
        path = self.package / 'README.md'
        path.unlink()
        path.symlink_to(self.package / 'evaluation_protocol.md')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.export()

    def test_output_must_be_new_ignored_results_directory(self):
        with self.assertRaises(ValueError):
            self.exporter.export_publication(self.root, self.root / 'publish')
        (self.root / '.gitignore').write_text('')
        with self.assertRaisesRegex(ValueError, 'ignored'):
            self.export()
        (self.root / '.gitignore').write_text('results/\n')
        self.output.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.export()

    def test_artifact_paths_and_aliases_cannot_escape(self):
        run = self.run_directory()
        with self.assertRaisesRegex(ValueError, 'alias'):
            self.export(runs={'../escape': run})
        legacy = self.root / 'evidence'
        legacy.mkdir()
        (legacy / 'report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'results'):
            self.export(runs={'legacy': legacy})
        private = self.root / 'results/private-planning-20261103'
        private.mkdir()
        (private / 'report.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.export(runs={'private': private})

    def test_malformed_json_rejected(self):
        (self.package / 'config.json').write_text('{broken')
        with self.assertRaisesRegex(ValueError, 'invalid JSON'):
            self.export()

    def test_regex_metadata_is_not_an_absolute_path(self):
        (self.package / 'config.json').write_text(json.dumps({'risk_filter': r'\b(?:unsafe)\b'}))
        self.export()

    def test_caller_private_terms_are_checked_even_in_source(self):
        (self.package / 'train.py').write_text('# ExamplePerson shared this\n')
        with self.assertRaisesRegex(ValueError, 'caller-supplied private term'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_deny_terms_file_is_local_only_and_must_be_ignored(self):
        private = self.root / 'results/deny-terms.json'
        private.parent.mkdir()
        private.write_text('["ExamplePerson", "ExampleAccount"]')
        terms = self.exporter.load_deny_terms(self.root, private)
        self.assertEqual(terms, ['ExamplePerson', 'ExampleAccount'])
        manifest = self.export(deny_terms=terms)
        self.assertTrue(manifest['caller_supplied_deny_terms_checked'])
        self.assertNotIn('ExamplePerson', json.dumps(manifest))
        self.assertFalse((self.output / 'results/deny-terms.json').exists())
        with self.assertRaises(ValueError):
            self.exporter.load_deny_terms(self.root, self.package / 'config.json')

    def test_plausible_secret_tokens_rejected_even_in_source(self):
        for prefix in ('hf_', 'ghp_', 'github_pat_', 'sk_', 'sk-proj-'):
            with self.subTest(prefix=prefix):
                (self.package / 'train.py').write_text('TOKEN = ' + repr(prefix + 'x' * 48) + '\n')
                with self.assertRaisesRegex(ValueError, 'credential'):
                    self.export()
                self.assertFalse(self.output.exists())

    def review_artifacts(self):
        pack = self.root / 'results/review-pack'
        pack.mkdir(parents=True)
        for lang in ('en', 'ko', 'zh'):
            (pack / f'review-{lang}.jsonl').write_text('{"response":"Synthetic response"}\n')
        (pack / 'evaluation_protocol.md').write_text('Scientific review protocol.\n')
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pack.iterdir()}
        payload = dict(schema_version=1, records={}, sources={}, config_sha256='a' * 64,
                       prepared_manifest_sha256='b' * 64)
        payload_bytes = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        manifest = dict(status='ready_for_blinded_review', files=files,
                        release_payload_sha256=hashlib.sha256(payload_bytes).hexdigest())
        (pack / 'manifest.json').write_text(json.dumps(manifest) + '\n')
        (pack / 'private').mkdir()
        (pack / 'private/mapping.json').write_text('{"answer_verbatim":"excluded"}\n')
        scores = self.root / 'results/review-scores'
        scores.mkdir()
        pack_hash = hashlib.sha256((pack / 'manifest.json').read_bytes()).hexdigest()
        review_hashes = {}
        for index in (1, 2):
            name = f'reviewer-{index:02d}'
            part = scores / 'reviews' / name / 'part-01.jsonl'
            part.parent.mkdir(parents=True)
            part.write_text('{"record_id":"example","notes":"Synthetic evidence"}\n')
            review_hashes[name] = [hashlib.sha256(part.read_bytes()).hexdigest()]
        key = dict(payload, released_after_review_validation=True, pack_manifest_sha256=pack_hash,
                   review_source_hashes=review_hashes)
        (scores / 'review-key.json').write_text(json.dumps(key) + '\n')
        for name in ('combined-reviews.jsonl', 'disagreements.jsonl'):
            (scores / name).write_text('')
        (scores / 'evaluation_protocol.md').write_bytes((pack / 'evaluation_protocol.md').read_bytes())
        report = dict(status='passed_provisional_review', pack_manifest_sha256=pack_hash,
                      review_source_hashes=review_hashes,
                      files={p.relative_to(scores).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in scores.rglob('*') if p.is_file()})
        (scores / 'report.json').write_text(json.dumps(report) + '\n')
        return pack, scores

    def test_review_artifacts_export_with_hashes_and_no_private_mapping(self):
        pack, scores = self.review_artifacts()
        manifest = self.export(review_packs={'study': pack}, review_scores={'study': scores})
        self.assertIn('review-packs/study/review-en.jsonl', manifest['files'])
        self.assertIn('review-scores/study/reviews/reviewer-02/part-01.jsonl', manifest['files'])
        self.assertFalse((self.output / 'review-packs/study/private/mapping.json').exists())

    def test_tampered_review_pack_rejected(self):
        pack, _ = self.review_artifacts()
        (pack / 'review-en.jsonl').write_text('{"response":"changed"}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.export(review_packs={'study': pack})

    def test_tampered_reviewer_part_rejected(self):
        pack, scores = self.review_artifacts()
        (scores / 'reviews/reviewer-01/part-01.jsonl').write_text('{"notes":"changed"}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.export(review_packs={'study': pack}, review_scores={'study': scores})

    def test_scores_require_their_matching_pack(self):
        _, scores = self.review_artifacts()
        with self.assertRaisesRegex(ValueError, 'matching review pack'):
            self.export(review_scores={'study': scores})


if __name__ == '__main__':
    unittest.main()
