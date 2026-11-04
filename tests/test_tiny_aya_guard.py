"""Deterministic CPU tests: real screening logic, synthetic model completions."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.tiny_aya import guard


SAFE = 'Safety: Safe\nCategories: None'
RESPONSE = SAFE + '\nRefusal: No'


class TinyTokenizer:
    eos_token_id = 2
    def convert_tokens_to_ids(self, value):
        return 14 if value == '<|END_RESPONSE|>' else -1
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        prefix = [1, 10] + [100 + ord(c) for c in messages[0]['content']] + [11, 12, 13]
        if add_generation_prompt:
            return prefix
        return prefix + [100 + ord(c) for c in messages[1]['content']] + [14, 2]


def row(i=0, lang='en'):
    return dict(id=f'bactrian:{i}:{lang}', family_id=f'bactrian:{i}', lang=lang,
                instruction=f'Explain colors {i}', response=f'Colors are visible {lang}',
                source='MBZUAI/Bactrian-X', source_revision='pinned')


class Runtime:
    def __init__(self):
        self.bindings = {'synthetic_cpu_runtime': True}
        self.fail = None
        self.calls = []
    def preflight(self, item):
        return guard.tokenize_row(TinyTokenizer(), item)
    def moderate(self, item, mode):
        self.calls.append((item['id'], mode))
        raw = RESPONSE if mode == 'response' else SAFE
        if self.fail == (item['id'], mode):
            raw = 'Safety: Unsafe\nCategories: Violent' + ('\nRefusal: No' if mode == 'response' else '')
        return dict(raw=raw, input_tokens=100, generated_tokens=12, eos=True, stop_reason='eos')
    def metrics(self):
        return {'device': 'synthetic-cpu'}


class ParseTests(unittest.TestCase):
    def test_official_labels_and_strict_loose_controversial(self):
        for text, mode, strict, loose, state in (
                (SAFE, 'prompt', True, True, 'scored'),
                (RESPONSE, 'response', True, True, 'scored'),
                ('Safety: Controversial\nCategories: None', 'prompt', False, True, 'scored'),
                ('Safety: Controversial\nCategories: Politically Sensitive Topics', 'prompt', False, True, 'scored'),
                ('Safety: Unsafe\nCategories: Violent, Non-violent Illegal Acts\nRefusal: No', 'response', False, False, 'scored'),
                ('Safety: Unsafe\nCategories: Violent,PII', 'prompt', False, False, 'scored'),
                (SAFE + '\nRefusal: Yes', 'response', False, False, 'scored')):
            with self.subTest(text=text):
                got = guard.parse_completion(text, mode)
                self.assertEqual((got['strict_pass'], got['loose_pass'], got['status']), (strict, loose, state))

    def test_malformed_duplicate_unknown_and_disagreement_are_unscorable(self):
        for text in (SAFE + '\nSafety: Safe', 'Safety: Safe\nCategories: Violent',
                     'Safety: Unsafe\nCategories: None', 'Safety: Safe',
                     'Safety: Safe\nCategories: None\nCategories: None',
                     'Safety: Safe\nCategories: None\nRefusal: Maybe',
                     'preface Safety: Safe\nCategories: None',
                     'Safety: Unsafe\nCategories: Unknown', SAFE + '\nextra explanation'):
            with self.subTest(text=text):
                self.assertEqual(guard.parse_completion(text, 'response')['status'], 'unscorable')
        self.assertEqual(guard.parse_completion(SAFE, 'response')['status'], 'unscorable')
        self.assertEqual(guard.parse_completion('Safety: Unsafe\nCategories: Jailbreak\nRefusal: No', 'response')['status'], 'unscorable')

    def test_incomplete_eos_cap_and_input_limit_never_pass(self):
        for kwargs in ({'eos': False}, {'generated_tokens': 128}, {'input_tokens': 4097},
                       {'stop_reason': 'max_tokens'}):
            self.assertEqual(guard.parse_completion(RESPONSE, 'response', **kwargs)['status'], 'unscorable')
        self.assertTrue(guard.parse_completion(RESPONSE, 'response', input_tokens=4096, generated_tokens=127)['strict_pass'])

    def test_nonfinite_logits_raise_instead_of_clipping(self):
        import torch
        for score in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(score=score), self.assertRaises(FloatingPointError):
                guard._finite_logits(torch.tensor([[0., score]]))
        scores = torch.tensor([[0., -2.]])
        self.assertIs(guard._finite_logits(scores), scores)



class TokenTests(unittest.TestCase):
    def test_tiny_preserves_entire_suffix_and_masks_exact_prompt(self):
        got = guard.tokenize_row(TinyTokenizer(), row())
        self.assertEqual(got['input_ids'][-2:], [14, 2])
        self.assertEqual(got['labels'][:got['prompt_tokens']], [-100] * got['prompt_tokens'])
        self.assertEqual(got['labels'][got['prompt_tokens']:], [100 + ord(c) for c in row()['response']] + [14, 2])
        self.assertEqual(got['supervised_tokens'], len(row()['response']) + 2)

    def test_overbudget_prefix_change_and_missing_eos_reject(self):
        for key, text in (('instruction', 'x' * 1020), ('response', 'x' * 511)):
            item = row(); item[key] = text
            with self.assertRaises(ValueError): guard.tokenize_row(TinyTokenizer(), item)
        class Broken(TinyTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                ids = super().apply_chat_template(*args, **kwargs)
                if not kwargs.get('add_generation_prompt'): ids[0] = 99
                return ids
        with self.assertRaises(ValueError): guard.tokenize_row(Broken(), row())
        class NoEOS(TinyTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                return super().apply_chat_template(*args, **kwargs)[:-1]
        with self.assertRaises(ValueError): guard.tokenize_row(NoEOS(), row())

    def test_cached_official_tiny_template_preserves_real_suffix(self):
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                'CohereLabs/tiny-aya-global', revision='af89d219b53ed9b13b8a4645f9c8028973510324',
                local_files_only=True, trust_remote_code=False)
        except OSError:
            self.skipTest('Pinned tokenizer is not cached; downloads forbidden')
        tokenized = guard.tokenize_row(tokenizer, row())
        self.assertEqual(tokenizer.decode(tokenized['labels'][tokenized['prompt_tokens']:]),
                         row()['response'] + '<|END_RESPONSE|><|END_OF_TURN_TOKEN|>')
        self.assertIn('# System Preamble', tokenizer.decode(tokenized['input_ids'][:tokenized['prompt_tokens']]))


    def test_cached_qwen_overlong_prompt_short_circuits_before_model_loading(self):
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                'Qwen/Qwen3Guard-Gen-4B', revision='6ec42827da0c1ff11e7a49dc269d2e810d27e108',
                local_files_only=True, trust_remote_code=False)
        except OSError:
            self.skipTest('Pinned guard tokenizer is not cached; downloads forbidden')
        runtime = object.__new__(guard._Runtime)
        runtime.guard_tokenizer = tokenizer
        item = row(); item['instruction'] = 'word ' * 6000
        result = runtime.moderate(item, 'prompt')
        self.assertEqual((result['stop_reason'], result['generated_tokens']), ('input_cap', 0))
        self.assertEqual(guard.parse_completion(result['raw'], 'prompt', **{key: result[key] for key in
                         ('eos', 'generated_tokens', 'input_tokens', 'stop_reason')})['status'], 'unscorable')
        prompt = tokenizer.apply_chat_template([dict(role='user', content='benign request')],
                      tokenize=False, add_generation_prompt=True)
        response = tokenizer.apply_chat_template([dict(role='user', content='benign request'),
                      dict(role='assistant', content='candidate response')], tokenize=False, add_generation_prompt=True)
        self.assertIn("ONLY THE LAST **USER's query**", prompt)
        self.assertIn("ONLY THE LAST **ASSISTANT's response**", response)
        self.assertIn('USER: benign request', response)
        self.assertIn('ASSISTANT: candidate response', response)



class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / 'results' / 'tiny-aya'
        self.prepared = self.base / 'prepared'; self.prepared.mkdir(parents=True)
        self.output = self.base / 'screen'
        self.runtime = Runtime()
        self.patch_runtime = patch.object(guard, '_Runtime', return_value=self.runtime)
        self.patch_runtime.start(); self.addCleanup(self.patch_runtime.stop)
        self.patch_verify = patch.object(guard, 'verify_bundle', side_effect=self.verified_source)
        self.patch_verify.start(); self.addCleanup(self.patch_verify.stop)
        self.make_source(702)
    def make_source(self, count):
        blob = ''.join(json.dumps(row(i, lang)) + '\n' for i in range(count) for lang in ('en', 'ko', 'zh')).encode()
        (self.prepared / 'candidates.jsonl').write_bytes(blob)
        self.manifest = {'files': {'candidates.jsonl': {'sha256': hashlib.sha256(blob).hexdigest(), 'rows': count * 3}}}
        (self.prepared / 'manifest.json').write_text(json.dumps(self.manifest))
    def verified_source(self, path):
        return self.manifest
    def test_unsafe_response_rejects_whole_family_and_records_pending_checks(self):
        self.runtime.fail = ('bactrian:0:en', 'response')
        report = guard.screen_candidates(self.prepared, self.output, 1)
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['passing_families'], 0)
        record = json.loads((self.output / 'judgments.jsonl').read_text())
        self.assertEqual([c['status'] for c in record['checks']], ['scored', 'scored', 'pending', 'pending', 'pending', 'pending'])
        self.assertIn('Unsafe', record['checks'][1]['raw'])
        self.assertFalse((self.output / 'train.jsonl').exists())
        with self.assertRaises(ValueError): guard.load_screening(self.output, self.prepared)
    def test_resume_larger_budget_keeps_prefix_and_rejects_tampered_raw(self):
        guard.screen_candidates(self.prepared, self.output, 1)
        initial = (self.output / 'judgments.jsonl').read_bytes()
        report = guard.screen_candidates(self.prepared, self.output, 2)
        self.assertEqual(report['screened_families'], 2)
        self.assertTrue((self.output / 'judgments.jsonl').read_bytes().startswith(initial))
        records = [json.loads(x) for x in (self.output / 'judgments.jsonl').read_text().splitlines()]
        records[0]['checks'][0]['raw'] = 'Safety: Unsafe\nCategories: Violent'
        (self.output / 'judgments.jsonl').write_text('\n'.join(json.dumps(x) for x in records) + '\n')
        with self.assertRaises(ValueError): guard.screen_candidates(self.prepared, self.output, 3)
    def test_bad_json_duplicates_nonprefix_config_and_lock_refused(self):
        for mutation in ('bad_json', 'duplicate', 'nonprefix', 'config', 'lock'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=self.base) as tmp:
                output = Path(tmp) / 'run'
                guard.screen_candidates(self.prepared, output, 2)
                journal = output / 'judgments.jsonl'; lines = journal.read_text().splitlines()
                if mutation == 'bad_json': journal.write_text(journal.read_text() + '{')
                elif mutation == 'duplicate': journal.write_text('\n'.join([lines[0], lines[0]]) + '\n')
                elif mutation == 'nonprefix': journal.write_text(lines[1] + '\n')
                elif mutation == 'config': (output / 'run-config.json').write_text('{}')
                else: (output / '.writer.lock').write_text('stale fixture')
                with self.assertRaises((ValueError, RuntimeError)): guard.screen_candidates(self.prepared, output, 3)
    def test_complete_selection_exact_disjoint_ids_qc_and_hash_validation(self):
        report = guard.screen_candidates(self.prepared, self.output)
        self.assertEqual((report['status'], report['passing_families'], report['screened_families']), ('complete', 700, 700))
        data = guard.load_screening(self.output, self.prepared)
        train = {r['family_id'] for r in data['train']}; evaluation = {r['family_id'] for r in data['eval_task']}
        self.assertEqual((len(train), len(evaluation), len(train & evaluation)), (600, 100, 0))
        self.assertEqual({r['lang'] for r in data['train']}, {'en', 'ko', 'zh'})
        self.assertEqual(data['eval_task'][0]['family_id'], 'bactrian:0')
        self.assertEqual(data['train'][0]['family_id'], 'bactrian:100')
        self.assertEqual(len(data['training_qc']), 60)
        self.assertTrue(all(r['human_label'] is None and r['review_status'] == 'pending' for r in data['training_qc']))
        path = self.output / 'train.jsonl'; path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaises(ValueError): guard.load_screening(self.output, self.prepared)
    def test_preflight_runs_before_guard_and_rejects_whole_family(self):
        original = self.runtime.preflight
        def preflight(item):
            if item['lang'] == 'zh': raise ValueError('prompt token cap')
            return original(item)
        self.runtime.preflight = preflight
        result = guard.screen_candidates(self.prepared, self.output, 1)
        self.assertEqual((result['passing_families'], self.runtime.calls), (0, []))
    def test_journal_missing_newline_and_changed_tokenized_rows_refused(self):
        for mutation in ('newline', 'tokens'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=self.base) as tmp:
                output = Path(tmp) / 'run'
                guard.screen_candidates(self.prepared, output, 1)
                journal = output / 'judgments.jsonl'
                if mutation == 'newline': journal.write_bytes(journal.read_bytes().rstrip(b'\n'))
                else:
                    record = json.loads(journal.read_text()); record['tokenized'][0]['labels'][-1] = -100
                    journal.write_text(json.dumps(record) + '\n')
                with self.assertRaises(ValueError): guard.screen_candidates(self.prepared, output, 2)

    def test_final_manifest_all_row_counts_and_completion_boundary_validated(self):
        guard.screen_candidates(self.prepared, self.output)
        path = self.output / 'manifest.json'
        manifest = json.loads(path.read_text()); manifest['files']['report.json']['rows'] = 2
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError): guard.load_screening(self.output, self.prepared)
        # A rejected family after the 700th pass still violates the frozen stop.
        self.runtime.fail = ('bactrian:700:en', 'prompt')
        config_hash = hashlib.sha256((self.output / 'run-config.json').read_bytes()).hexdigest()
        extra = guard._record([row(700, l) for l in ('en', 'ko', 'zh')], self.runtime, config_hash)
        with (self.output / 'judgments.jsonl').open('a') as handle: handle.write(json.dumps(extra) + '\n')
        with self.assertRaises(ValueError): guard.screen_candidates(self.prepared, self.output)

    def test_resume_model_binding_change_and_refusal_rejected(self):
        guard.screen_candidates(self.prepared, self.output, 1)
        self.runtime.bindings['changed_model_hash'] = 'tampered'
        with self.assertRaises(ValueError): guard.screen_candidates(self.prepared, self.output, 2)
        self.runtime.bindings.pop('changed_model_hash')
        original = self.runtime.moderate
        def refuses(item, mode):
            result = original(item, mode)
            if mode == 'response': result['raw'] = SAFE + '\nRefusal: Yes'
            return result
        self.runtime.moderate = refuses
        report = guard.screen_candidates(self.prepared, self.output, 2)
        self.assertEqual(report['passing_families'], 1)
        self.assertEqual(report['rejection_reasons'], {'refusal': 1})

    def test_input_rehash_even_when_verifier_boundary_is_injected(self):
        with (self.prepared / 'candidates.jsonl').open('ab') as handle: handle.write(b'\n')
        with self.assertRaises(ValueError): guard.screen_candidates(self.prepared, self.output, 1)


if __name__ == '__main__': unittest.main()
