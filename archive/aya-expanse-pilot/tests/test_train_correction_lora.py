"""Integrity and scheduling tests; no model weights or GPU operations."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tests.test_prepare_corrections import FakeTokenizer, POLICY, V2, V3, family, utility_family, v2_config

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'evidence/correction-preparation/preflight'


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module('scripts.train_correction_lora')
        except ModuleNotFoundError as exc:
            self.fail(f'Training runner is missing: {exc}')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.prepared = Path(self.temp.name) / 'prepared'
        shutil.copytree(ARCHIVE, self.prepared)
        # This temporary archive represents fresh preparation with unchanged v1
        # bytes; the historical on-disk archive and its old source hash stay intact.
        mpath = self.prepared / 'manifest.json'; manifest = json.loads(mpath.read_text())
        manifest['script_sha256'] = hashlib.sha256((ROOT / 'scripts/prepare_corrections.py').read_bytes()).hexdigest()
        mpath.write_text(json.dumps(manifest))

    def versioned_archive(self, version=2):
        root = Path(self.temp.name) / f'v{version}-root'; root.mkdir()
        config = v2_config(json.loads((ROOT / 'configs/correction-lora.json').read_text()),
                           experiment_id={2: V2, 3: V3}[version])
        policy_bytes = json.dumps(POLICY).encode()
        config['policy_sha256'] = hashlib.sha256(policy_bytes).hexdigest()
        rows = [r for i in range(16) for r in family(f'train_{i}')]
        rows += [r for i in range(8) for r in utility_family(f'utility_{i}')]
        rows += [r for i in range(4) for r in family(f'cal_{i}', 'calibration')]
        contents = {
            f'data/corrections-v{version}/examples.jsonl': ''.join(json.dumps(r) + '\n' for r in rows).encode(),
            f'configs/correction-lora-v{version}.json': json.dumps(config).encode(),
            'data/pilot/policy.json': policy_bytes,
            'data/pilot/scenarios.jsonl': b'',
            'configs/runtime.json': (ROOT / 'configs/runtime.json').read_bytes(),
            'configs/tools.lock.json': (ROOT / 'configs/tools.lock.json').read_bytes(),
            'scripts/prepare_corrections.py': (ROOT / 'scripts/prepare_corrections.py').read_bytes(),
        }
        for name, content in contents.items():
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        prepare = importlib.import_module('scripts.prepare_corrections')
        output = root / 'preflight'
        template = 'fixture template'
        metadata = {'actual_snapshot_commit': config['model_revision'], 'eos_token_id': 2,
                    'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest()}
        with patch.object(prepare, 'ROOT', root), patch.object(prepare, 'load_cached_tokenizer',
                return_value=(FakeTokenizer(), template, metadata)):
            self.assertEqual(prepare.main(['--config', str(root / f'configs/correction-lora-v{version}.json'),
                                          '--output', str(output)]), 0)
        return root, output

    def rewrite(self, filename, rows):
        path = self.prepared / filename
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        manifest_path = self.prepared / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files'][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    def records(self, filename):
        return [json.loads(line) for line in (self.prepared / filename).read_text().splitlines()]

    def test_accepts_current_frozen_archive(self):
        data = self.api.load_prepared(self.prepared)
        self.assertEqual(len(data['train']), 32)
        self.assertEqual(len(data['calibration']), 16)
        self.assertEqual(data['config']['policy_sha256'], data['manifest']['inputs']['policy']['sha256'])

    def test_prior_source_hash_cannot_be_silently_reused(self):
        path = self.prepared / 'manifest.json'; manifest = json.loads(path.read_text())
        manifest['script_sha256'] = '0' * 64; path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'implementation hash'):
            self.api.load_prepared(self.prepared)

    def test_versioned_archive_loads_only_its_canonical_inputs(self):
        for version in (2, 3):
            with self.subTest(version=version):
                root, archive = self.versioned_archive(version)
                data = self.api.load_prepared(archive, root=root)
                self.assertEqual(len(data['train']), 80)
                self.assertEqual(len(data['calibration']), 16)
                self.assertEqual(data['config']['epochs'], 2)
                self.assertEqual(data['config']['train_steps'], 160)
                self.assertEqual(sum(r['category'] == 'utility' for r in data['train']), 16)

    def test_versioned_rogue_manifest_path_is_rejected_even_with_identical_bytes(self):
        for version in (2, 3):
            with self.subTest(version=version):
                root, archive = self.versioned_archive(version)
                rogue = root / 'rogue.jsonl'; shutil.copyfile(root / f'data/corrections-v{version}/examples.jsonl', rogue)
                path = archive / 'manifest.json'; manifest = json.loads(path.read_text())
                manifest['inputs']['data']['path'] = str(rogue); path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'canonical.*path'):
                    self.api.load_prepared(archive, root=root)

    def test_v2_manifest_cannot_redirect_to_stale_copy_when_canonical_data_changes(self):
        root, archive = self.versioned_archive()
        path = root / 'data/corrections-v2/examples.jsonl'; path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'canonical input hash'):
            self.api.load_prepared(archive, root=root)

    def test_rejects_unhashed_edit(self):
        with (self.prepared / 'train.jsonl').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.api.load_prepared(self.prepared)

    def test_rejects_calibration_row_in_optimizer_partition_even_if_rehashed(self):
        train, calibration = self.records('train.jsonl'), self.records('calibration.jsonl')
        train[0] = calibration[0]
        self.rewrite('train.jsonl', train)
        with self.assertRaises(ValueError):
            self.api.load_prepared(self.prepared)

    def test_rejects_duplicate_training_id_even_if_rehashed(self):
        rows = self.records('train.jsonl'); rows[1] = deepcopy(rows[0]); self.rewrite('train.jsonl', rows)
        with self.assertRaises(ValueError):
            self.api.load_prepared(self.prepared)

    def test_rejects_unmasked_policy_token_even_if_rehashed(self):
        rows = self.records('train.jsonl'); rows[0]['labels'][0] = rows[0]['input_ids'][0]; self.rewrite('train.jsonl', rows)
        with self.assertRaises(ValueError):
            self.api.load_prepared(self.prepared)

    def test_rejects_missing_eos_even_if_rehashed(self):
        rows = self.records('train.jsonl'); rows[0]['input_ids'][-1] = 12; rows[0]['labels'][-1] = 12; self.rewrite('train.jsonl', rows)
        with self.assertRaisesRegex(ValueError, 'EOS'):
            self.api.load_prepared(self.prepared)

    def test_rejects_mask_count_or_attention_mismatch(self):
        for field, value in [('supervised_tokens', 0), ('attention_mask', [1])]:
            with self.subTest(field=field):
                rows = self.records('train.jsonl'); rows[0][field] = value; self.rewrite('train.jsonl', rows)
                with self.assertRaises(ValueError):
                    self.api.load_prepared(self.prepared)

    def test_rejects_archive_file_path_traversal(self):
        path = self.prepared / 'manifest.json'; manifest = json.loads(path.read_text())
        manifest['files']['../outside.json'] = '0' * 64; path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.api.load_prepared(self.prepared)

    def test_rejects_rebound_policy_or_config(self):
        config = json.loads((self.prepared / 'config.json').read_text()); config['learning_rate'] = 99
        path = self.prepared / 'config.json'; path.write_text(json.dumps(config))
        mpath = self.prepared / 'manifest.json'; manifest = json.loads(mpath.read_text())
        manifest['files']['config.json'] = hashlib.sha256(path.read_bytes()).hexdigest(); mpath.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.api.load_prepared(self.prepared)

    def test_fixed_config_rejects_unsupported_hyperparameters(self):
        config = json.loads((self.prepared / 'config.json').read_text())
        self.api.validate_training_config(config)
        for field, value in [('learning_rate', 0.5), ('gradient_checkpointing', False), ('seed', True), ('target_modules', ['q_proj']), ('save', 'merged')]:
            with self.subTest(field=field):
                invalid = deepcopy(config); invalid[field] = value
                with self.assertRaises(ValueError):
                    self.api.validate_training_config(invalid)

    def test_expanded_experiments_require_exact_schedule_and_learning_rate(self):
        for experiment_id in (V2, V3):
            with self.subTest(experiment_id=experiment_id):
                config = v2_config(json.loads((self.prepared / 'config.json').read_text()), experiment_id=experiment_id)
                self.api.validate_training_config(config)
                for field, value in [('epochs', 1), ('epochs', True), ('train_steps', 80), ('train_steps', 159),
                                     ('learning_rate', 0.0001), ('experiment_id', '../unknown'),
                                     ('expected_rows', {'train': 32, 'calibration': 16})]:
                    with self.subTest(field=field, value=value):
                        invalid = deepcopy(config); invalid[field] = value
                        with self.assertRaises(ValueError):
                            self.api.validate_training_config(invalid)

    def test_two_epochs_reshuffle_all_training_ids_exactly_once_per_epoch(self):
        rows = [{'id': str(i), 'split': 'train'} for i in range(80)]
        order = self.api.training_order(rows, seed=42, steps=160, epochs=2)
        self.assertEqual(order, self.api.training_order(rows, seed=42, steps=160, epochs=2))
        self.assertEqual(len(order), 160)
        self.assertCountEqual(order[:80], [str(i) for i in range(80)])
        self.assertCountEqual(order[80:], [str(i) for i in range(80)])
        self.assertNotEqual(order[:80], order[80:])
        self.assertEqual(order[:80], self.api.training_order(rows, seed=42, steps=80))

    def test_declared_epochs_reject_partial_extra_and_noninteger_schedules(self):
        rows = [{'id': str(i), 'split': 'train'} for i in range(80)]
        for steps, epochs in ((159, 2), (80, 2), (160, 1), (240, 2), (160, True), (160, 2.0), (160, 0)):
            with self.subTest(steps=steps, epochs=epochs), self.assertRaises(ValueError):
                self.api.training_order(rows, seed=42, steps=steps, epochs=epochs)
        rows[-1]['split'] = 'calibration'
        with self.assertRaises(ValueError):
            self.api.training_order(rows, seed=42, steps=160, epochs=2)

    def test_deterministic_order_is_one_complete_train_only_epoch(self):
        data = self.api.load_prepared(self.prepared)
        first = self.api.training_order(data['train'], seed=42, steps=32)
        self.assertEqual(first, self.api.training_order(data['train'], seed=42, steps=32))
        self.assertEqual(set(first), {r['id'] for r in data['train']})
        self.assertEqual(len(first), len(set(first)))
        self.assertNotEqual(first, self.api.training_order(data['train'], seed=7, steps=32))
        self.assertTrue(set(first).isdisjoint({r['id'] for r in data['calibration']}))

    def test_order_rejects_partial_epoch_or_calibration(self):
        data = self.api.load_prepared(self.prepared)
        with self.assertRaises(ValueError): self.api.training_order(data['train'], seed=42, steps=24)
        with self.assertRaises(ValueError): self.api.training_order(data['calibration'], seed=42, steps=16)
        with self.assertRaises(ValueError): self.api.training_order(data['train'], seed=True, steps=32)

    def test_calibration_is_token_weighted_not_row_weighted(self):
        result = self.api.aggregate_calibration([{'id': 'a', 'loss': 1.0, 'supervised_tokens': 2}, {'id': 'b', 'loss': 3.0, 'supervised_tokens': 6}])
        self.assertEqual(result['token_weighted_loss'], 2.5)
        self.assertEqual(result['supervised_tokens'], 8)
        self.assertEqual(result['rows'], 2)

    def test_calibration_rejects_nonfinite_zero_duplicate_or_empty(self):
        for rows in [[], [{'id': 'a', 'loss': float('nan'), 'supervised_tokens': 2}], [{'id': 'a', 'loss': 1, 'supervised_tokens': 0}], [{'id': 'a', 'loss': -1, 'supervised_tokens': 2}], [{'id': 'a', 'loss': 1, 'supervised_tokens': 1}] * 2]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.api.aggregate_calibration(rows)

    def test_prepare_only_cli_never_imports_training_runtime(self):
        output = Path(self.temp.name) / 'result'
        with patch.object(self.api, 'verify_retokenized', return_value=(object(), {'scope': 'mock tokenizer'})), patch.dict('sys.modules', {'torch': None, 'safetune': None}):
            code = self.api.main(['--prepared', str(self.prepared), '--output', str(output), '--prepare-only'])
        self.assertEqual(code, 0)
        report = json.loads((output / 'report.json').read_text())
        self.assertEqual(report['status'], 'prepared_not_trained')
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertFalse(report['model_weights_loaded'])
        self.assertEqual(len(report['training_order']), 32)
        self.assertEqual(len(report['calibration_example_ids']), 16)
        self.assertEqual((output / 'training-config.json').read_bytes(), (self.prepared / 'config.json').read_bytes())

    def test_v2_prepare_only_cli_reports_two_complete_epochs_without_runtime(self):
        root, archive = self.versioned_archive(); output = root / 'result'
        with patch.object(self.api, 'ROOT', root), patch.object(self.api, 'verify_retokenized',
                return_value=(object(), {'scope': 'mock tokenizer'})), patch.dict('sys.modules', {'torch': None, 'safetune': None}):
            code = self.api.main(['--prepared', str(archive), '--output', str(output), '--prepare-only'])
        self.assertEqual(code, 0)
        report = json.loads((output / 'report.json').read_text())
        self.assertEqual(report['status'], 'prepared_not_trained')
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertFalse(report['model_weights_loaded'])
        self.assertEqual(len(report['training_order']), 160)
        self.assertEqual(len(set(report['training_order'][:80])), 80)
        self.assertEqual(len(set(report['training_order'][80:])), 80)
        self.assertEqual(len(report['calibration_example_ids']), 16)
        self.assertTrue(set(report['training_order']).isdisjoint(report['calibration_example_ids']))

    def test_prepare_only_refuses_existing_output(self):
        with self.assertRaises(SystemExit):
            self.api.main(['--prepared', str(self.prepared), '--output', str(self.prepared), '--prepare-only'])
        self.assertTrue((self.prepared / 'manifest.json').exists())

    def test_retokenization_rejects_changed_tokenizer_files(self):
        data = self.api.load_prepared(self.prepared)
        metadata = deepcopy(data['manifest']['tokenizer']); metadata['files']['tokenizer.json'] = 'f' * 64
        with patch.object(self.api, 'load_cached_tokenizer', return_value=(object(), '', metadata)):
            with self.assertRaisesRegex(ValueError, 'tokenizer'):
                self.api.verify_retokenized(data)

    def test_retokenization_rejects_self_consistent_changed_token_ids(self):
        data = self.api.load_prepared(self.prepared)
        originals = {r['id']: {k: r[k] for k in ('id','input_ids','attention_mask','labels','prompt_tokens','sequence_tokens','supervised_tokens')} for r in data['train'] + data['calibration']}
        data['train'][0]['input_ids'] = list(data['train'][0]['input_ids']); data['train'][0]['input_ids'][0] += 1
        metadata = deepcopy(data['manifest']['tokenizer'])
        template = (self.prepared / 'chat-template.txt').read_text()
        with patch.object(self.api, 'load_cached_tokenizer', return_value=(object(), template, metadata)), patch.object(self.api, 'prepare_example', side_effect=lambda row, *args: originals[row['id']]):
            with self.assertRaisesRegex(ValueError, 'retoken'):
                self.api.verify_retokenized(data)


if __name__ == '__main__':
    unittest.main()
