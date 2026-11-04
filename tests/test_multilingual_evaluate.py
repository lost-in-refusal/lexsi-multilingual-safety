"""CPU-only checks for the frozen multilingual generation boundary."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.multilingual_repair import evaluate as ev


def blob(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / 'prepared'
        self.prepared.mkdir()
        self.config = {
            'model': ev.MODEL, 'model_revision': ev.MODEL_REVISION, 'seed': 42,
            'dtype': 'bfloat16', 'device': 'mps',
            'attention_implementation': 'eager',
            'training': {'epochs': 1, 'learning_rate': 1e-5, 'lora_r': 4, 'lora_alpha': 8,
                         'lora_dropout': 0.0, 'target_modules': sorted(ev.TARGET_MODULES),
                         'max_sequence_tokens': 1024, 'mps_memory_fraction': 0.8, 'rho': 1.0},
            'generation': {'temperature': 0.0, 'max_new_tokens': 256},
        }
        self.task = dict(id='task-1', source_id='public:1', split='development', language='ko',
                         category='task', request='질문', reference='정답', system='system',
                         input_ids=[11, 12, 13, 20, 21, 2], attention_mask=[1] * 6,
                         labels=[-100] * 3 + [20, 21, 2], prompt_tokens=3,
                         sequence_tokens=6, supervised_tokens=3)
        self.harm = dict(id='safety-1', source_id='public:2', split='development', language='en',
                         category='disallowed', request='request', system='system',
                         input_ids=[11, 14, 13], attention_mask=[1] * 3,
                         prompt_tokens=3, sequence_tokens=3)
        values = {'config.json': blob(self.config)}
        for split in ('train', 'calibration', 'development', 'confirmation', 'alignment'):
            rows = [dict(self.task, split=split), dict(self.harm, split=split)]
            if split == 'alignment':
                rows = [dict(self.harm, split=split)]
            values[split + '.jsonl'] = ''.join(json.dumps(r) + '\n' for r in rows).encode()
        for name, data in values.items():
            (self.prepared / name).write_bytes(data)
        self.manifest = {'status': 'passed_tokenization_preflight', 'truncated_examples': 0,
                         'files': {name: digest(data) for name, data in values.items()},
                         'tokenizer': {'actual_snapshot_commit': ev.MODEL_REVISION,
                                       'eos_token_id': 2, 'chat_template_sha256': 'a' * 64},
                         'sources': {'task': {'repo': 'public/example', 'revision': 'b' * 40,
                                              'source_sha256': 'c' * 64, 'license': 'test'}}}
        (self.prepared / 'manifest.json').write_bytes(blob(self.manifest))

    def bundle(self):
        return ev.load_prepared(self.prepared, 'development')

    def training_run(self):
        run = self.root / 'trained'
        adapter = run / 'adapter'
        adapter.mkdir(parents=True)
        adapter_config = {'base_model_name_or_path': ev.MODEL, 'revision': ev.MODEL_REVISION,
                          'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 4,
                          'lora_alpha': 8, 'lora_dropout': 0.0, 'bias': 'none',
                          'target_modules': sorted(ev.TARGET_MODULES), 'inference_mode': True}
        (adapter / 'adapter_config.json').write_bytes(blob(adapter_config))
        (adapter / 'adapter_model.safetensors').write_bytes(b'test-only-adapter')
        (run / 'training-config.json').write_bytes(blob(self.config))
        (run / 'prepared-manifest.json').write_bytes(blob(self.manifest))
        report = {'status': 'passed_training', 'arm': 'plain', 'run_kind': 'full_epoch',
                  'config_sha256': digest(blob(self.config)),
                  'prepared_manifest_sha256': digest(blob(self.manifest)),
                  'prepared_file_hashes': self.manifest['files'],
                  'training_script_sha256': 'd' * 64,
                  'expected_steps': 2, 'optimizer_steps': 2,
                  'training_order': ['task-1', 'safety-1'],
                  'adapter_directory': 'adapter', 'adapter_files': {
                      p.name: digest(p.read_bytes()) for p in adapter.iterdir()},
                  'saved_adapter_reload_exact': True, 'all_base_parameters_frozen': True,
                  'frozen_parameter_samples_unchanged': True,
                  'model': ev.MODEL, 'model_revision': ev.MODEL_REVISION,
                  'attention_implementation': 'eager',
                  'training_config': self.config}
        (run / 'report.json').write_bytes(blob(report))
        return run, report

    def test_manifest_hash_tamper_rejected_before_generation(self):
        (self.prepared / 'development.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.bundle()

    def test_manifest_path_traversal_rejected(self):
        self.manifest['files']['../outside.json'] = '0' * 64
        (self.prepared / 'manifest.json').write_bytes(blob(self.manifest))
        with self.assertRaisesRegex(ValueError, 'file names'):
            self.bundle()

    def test_confirmation_disabled_at_api_and_cli(self):
        with self.assertRaisesRegex(ValueError, 'confirmation'):
            ev.load_prepared(self.prepared, 'confirmation')
        with self.assertRaises(SystemExit) as caught:
            ev.main(['--prepared', str(self.prepared), '--output', str(self.root / 'output'),
                     '--split', 'confirmation'])
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse((self.root / 'output').exists())

    def test_task_reference_is_never_in_generation_input(self):
        row = self.bundle()['rows'][0]
        self.assertEqual(ev.prompt_inputs(row), ([11, 12, 13], [1, 1, 1]))
        self.assertEqual(ev.prompt_inputs(self.harm), ([11, 14, 13], [1, 1, 1]))

    def test_task_missing_prompt_boundary_rejected(self):
        row = copy.deepcopy(self.task)
        del row['prompt_tokens']
        with self.assertRaisesRegex(ValueError, 'prompt'):
            ev.prompt_inputs(row)

    def test_generate_passes_only_prompt_ids_and_preserves_raw_tokens(self):
        import torch

        class FakeModel:
            def parameters(self):
                yield torch.nn.Parameter(torch.zeros(1))

            def generate(self, **kwargs):
                self.kwargs = kwargs
                self.inference_mode = torch.is_inference_mode_enabled()
                return torch.tensor([[11, 12, 13, 31, 2]])

        class FakeTokenizer:
            pad_token_id = 2

            def decode(self, ids, skip_special_tokens):
                self.decoded_ids = ids
                return '  답변\n'

        model, tokenizer = FakeModel(), FakeTokenizer()
        record = ev.generate_one(model, tokenizer, self.task, max_new_tokens=2, eos_token_ids=[2])
        self.assertEqual(model.kwargs['input_ids'].tolist(), [[11, 12, 13]])
        self.assertEqual(model.kwargs['attention_mask'].tolist(), [[1, 1, 1]])
        self.assertFalse(model.kwargs['do_sample'])
        self.assertEqual(model.kwargs['num_beams'], 1)
        self.assertTrue(model.inference_mode)
        self.assertEqual(record['generated_token_ids'], [31, 2])
        self.assertEqual(record['response'], '  답변\n')
        self.assertEqual(record['response_sha256'], digest('  답변\n'.encode()))
        self.assertEqual(record['finish_reason'], 'eos')
        self.assertFalse(record['cap_reached'])

    def test_prompt_retokenization_rejects_missing_system_or_boundary_change(self):
        # No weights are loaded: patch the offline tokenizer boundary itself.
        class FakeTokenizer:
            eos_token_id = 2
            pad_token_id = 2

            def get_chat_template(self):
                return 'template'

            def apply_chat_template(self, messages, **kwargs):
                return [999]

        bundle = self.bundle()
        bundle['manifest']['tokenizer']['chat_template_sha256'] = digest(b'template')
        with patch('huggingface_hub.snapshot_download', return_value='/cache/' + ev.MODEL_REVISION), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=FakeTokenizer()), \
             self.assertRaisesRegex(ValueError, 'prompt token verification'):
            ev.load_cached_tokenizer(bundle)

    def test_eos_takes_precedence_at_cap(self):
        self.assertEqual(ev.termination([10, 2], [2], 2),
                         {'ended_with_eos': True, 'cap_reached': False, 'finish_reason': 'eos'})
        self.assertEqual(ev.termination([10, 11], [2], 2),
                         {'ended_with_eos': False, 'cap_reached': True, 'finish_reason': 'token_cap'})
        self.assertEqual(ev.termination([10], [2], 2)['finish_reason'], 'other')
        self.assertEqual(ev.termination([], [2], 2)['finish_reason'], 'empty_generation')

    def test_complete_training_identity_accepted(self):
        run, report = self.training_run()
        result = ev.load_training_run(run, self.bundle())
        self.assertEqual(result['identity']['adapter_files'], report['adapter_files'])
        self.assertEqual(result['identity']['arm'], 'plain')

    def test_incomplete_training_and_failed_checks_rejected(self):
        run, report = self.training_run()
        for field, value in [('status', 'running'), ('optimizer_steps', 1),
                             ('saved_adapter_reload_exact', False), ('model_revision', 'e' * 40),
                             ('attention_implementation', 'sdpa'), ('attention_implementation', None)]:
            broken = dict(report, **{field: value})
            (run / 'report.json').write_bytes(blob(broken))
            with self.subTest(field=field), self.assertRaises(ValueError):
                ev.load_training_run(run, self.bundle())

    def test_completed_feasibility_training_is_always_labeled(self):
        run, report = self.training_run()
        report.update(status='passed_feasibility', run_kind='feasibility', expected_steps=1,
                      optimizer_steps=1, training_order=['task-1'])
        (run / 'report.json').write_bytes(blob(report))
        self.assertTrue(ev.load_training_run(run, self.bundle())['identity']['feasibility_only'])

    def test_adapter_tamper_and_extra_file_rejected(self):
        run, report = self.training_run()
        (run / 'adapter/adapter_model.safetensors').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            ev.load_training_run(run, self.bundle())
        (run / 'adapter/adapter_model.safetensors').write_bytes(b'test-only-adapter')
        (run / 'adapter/unexpected.bin').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'files differ'):
            ev.load_training_run(run, self.bundle())

    def test_greedy_seed_and_new_output_are_enforced(self):
        for field, value in [('seed', 1), ('model_revision', 'e' * 40)]:
            config = copy.deepcopy(self.config)
            config[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ev.validate_config(config)
        config = copy.deepcopy(self.config)
        config['generation']['temperature'] = 0.7
        with self.assertRaises(ValueError):
            ev.validate_config(config)
        with self.assertRaises(SystemExit):
            ev.main(['--prepared', str(self.prepared), '--output', str(self.prepared)])

    def test_config_requires_explicit_eager_attention(self):
        for value in (None, 'sdpa', 'flash_attention_2'):
            config = copy.deepcopy(self.config)
            config['attention_implementation'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'attention'):
                ev.validate_config(config)

    def test_attention_backend_is_set_and_verified_before_forward(self):
        class FakeModel:
            config = SimpleNamespace(_attn_implementation='sdpa')

            def set_attn_implementation(self, implementation):
                self.config._attn_implementation = implementation

            def named_modules(self):
                return [('model.layers.0.self_attn', SimpleNamespace(config=self.config))]

        model = FakeModel()
        self.assertEqual(ev.configure_attention(model, self.config), 'eager')
        self.assertEqual(ev.verify_attention(model, self.config), 'eager')
        model.config._attn_implementation = 'sdpa'
        with self.assertRaisesRegex(ValueError, 'attention'):
            ev.verify_attention(model, self.config)

    def test_ignored_attention_selection_is_rejected(self):
        model = SimpleNamespace(config=SimpleNamespace(_attn_implementation='sdpa'),
                                set_attn_implementation=lambda implementation: None)
        with self.assertRaisesRegex(ValueError, 'attention'):
            ev.configure_attention(model, self.config)

    def test_decoder_attention_mismatch_is_rejected(self):
        model = SimpleNamespace(config=SimpleNamespace(_attn_implementation='eager'),
            named_modules=lambda: [('model.layers.0.self_attn',
                                    SimpleNamespace(config=SimpleNamespace(_attn_implementation='sdpa')))])
        with self.assertRaisesRegex(ValueError, 'Decoder attention'):
            ev.verify_attention(model, self.config)

    def test_real_auditkit_precomputed_ingestion_preserves_raw_outputs(self):
        rows = [self.task, self.harm]
        records = [{'sample_id': 'task-1', 'response': '대답'},
                   {'sample_id': 'safety-1', 'response': '  '}]
        output = self.root / 'auditkit-result.json'
        result = ev.ingest_auditkit(rows, records, output, {'run_nonce': 'test-' + self.root.name})
        self.assertEqual(result['status'], 'passed_diagnostic_ingestion')
        self.assertEqual(result['nonempty_output'], 0.5)
        saved = json.loads(output.read_text())
        self.assertTrue(saved['metadata']['diagnostic_only'])
        self.assertEqual(saved['metadata']['semantic_review'], 'pending')


if __name__ == '__main__':
    unittest.main()
