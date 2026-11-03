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
