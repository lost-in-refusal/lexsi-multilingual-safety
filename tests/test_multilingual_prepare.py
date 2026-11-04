import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'experiments/multilingual_repair/prepare.py'
spec = importlib.util.spec_from_file_location('multilingual_prepare', PATH)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class TinyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = messages[0]['content'] + '|' + messages[1]['content'] + '|assistant|'
        ids = [ord(c) for c in text]
        return ids if add_generation_prompt else ids + [ord(c) for c in messages[-1]['content']] + [0]

    def decode(self, ids, **kwargs):
        return ''.join(chr(c) for c in ids)


class PreparationTests(unittest.TestCase):
    def test_assistant_only_mask_and_terminal_eos(self):
        row = {'system': 's', 'request': 'q', 'reference': 'answer'}
        out = prep.prepare_tokens(row, TinyTokenizer(), 100)
        n = out['prompt_tokens']
        self.assertEqual(out['labels'][:n], [-100] * n)
        self.assertEqual(out['labels'][n:], [ord(c) for c in 'answer'] + [0])
        self.assertEqual(out['supervised_tokens'], 7)

    def test_long_sequence_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, 'truncation'):
            prep.prepare_tokens({'system': 's', 'request': 'q', 'reference': 'a' * 100}, TinyTokenizer(), 25)

    def test_alignment_has_no_target_or_labels(self):
        out = prep.prepare_tokens({'system': 's', 'request': 'q'}, TinyTokenizer(), 100)
        self.assertNotIn('labels', out)
        self.assertEqual(out['prompt_tokens'], len(out['input_ids']))

    def test_normalized_cross_split_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate prompt'):
            prep.assert_disjoint({'train': [{'id': '1', 'request': 'HELLO  world'}],
                                  'confirmation': [{'id': '2', 'request': 'hello world'}]})

    def test_translations_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, 'family crosses'):
            prep.assert_disjoint({'alignment': [{'id': '1', 'request': 'hello', 'family_id': 'f'}],
                                  'development': [{'id': '2', 'request': '안녕', 'family_id': 'f'}]})

    def test_partition_is_order_independent_and_unique(self):
        rows = [{'id': f'{lang}{i}', 'source_id': f'{lang}{i}', 'language': lang, 'request': f'{lang} question {i}'}
                for lang in ('en', 'ko', 'zh') for i in range(8)]
        cfg = {'seed': 42, 'data': {'train_counts': dict(en=3, ko=3, zh=3),
                                  'confirmation_per_language': 2, 'development_per_language': 1, 'calibration_per_language': 1}}
        a = prep.partition_tasks(rows, cfg)
        self.assertEqual(a, prep.partition_tasks(list(reversed(rows)), cfg))
        self.assertEqual([len(a[x]) for x in ('train', 'calibration', 'development', 'confirmation')], [9, 3, 3, 6])


if __name__ == '__main__':
    unittest.main()
