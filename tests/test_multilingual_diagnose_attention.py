"""CPU-only checks for portable attention-diagnostic statistics and data access."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.multilingual_repair import diagnose_attention as diagnostic
from experiments.multilingual_repair import train
from safetune.core.optim.safegrad import safegrad_kl_alignment_loss


class AttentionDiagnosticTests(unittest.TestCase):
    def test_comparison_uses_reference_to_candidate_batchmean_without_masking(self):
        reference = torch.tensor([[[1., 2., -1.], [3., -2., 1.]]])
        candidate = torch.tensor([[[2., -1., 0.], [1., 2., 3.]]], requires_grad=True)
        actual = diagnostic.compare_logits(reference, candidate, safegrad_kl_alignment_loss)
        expected = (reference.softmax(-1) * (reference.log_softmax(-1) - candidate.log_softmax(-1))).sum()
        self.assertFalse(actual['equal'])
        self.assertAlmostEqual(actual['kl_batchmean'], expected.item(), places=6)
        self.assertEqual(actual['max_abs'], 4.)
        self.assertIsNone(candidate.grad)
        same = diagnostic.compare_logits(reference, reference, safegrad_kl_alignment_loss)
        self.assertTrue(same['equal'])
        self.assertEqual(same['kl_batchmean'], safegrad_kl_alignment_loss(reference, reference).item())

    def test_all_four_modes_required_and_nonfinite_logits_rejected(self):
        value = torch.ones((1, 2, 3))
        with self.assertRaisesRegex(ValueError, 'four execution modes'):
            diagnostic.compare_modes({'disabled_nograd': value}, safegrad_kl_alignment_loss)
        result = diagnostic.compare_modes({mode: value for mode in diagnostic.MODES}, safegrad_kl_alignment_loss)
        self.assertTrue(result['active_grad_vs_nograd_equal'])
        for nonfinite in (float('nan'), float('inf')):
            with self.subTest(nonfinite=nonfinite), self.assertRaisesRegex(ValueError, 'Nonfinite diagnostic logits'):
                diagnostic.compare_logits(value, value * nonfinite, safegrad_kl_alignment_loss)

    def test_only_alignment_partition_is_read_and_its_bytes_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = (Path(__file__).resolve().parents[1] / 'experiments/multilingual_repair/config.json').read_bytes()
            rows = [dict(id=f'{language}-{number}', source_id=f'public-{language}-{number}',
                         split='alignment', category='disallowed', language=language,
                         request='synthetic diagnostic fixture', system='test',
                         input_ids=[1, 2], attention_mask=[1, 1], sequence_tokens=2, prompt_tokens=2)
                    for language in ('en', 'ko', 'zh') for number in range(24)]
            alignment = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            files = {name: '0' * 64 for name in train.REQUIRED_FILES}
            files.update({'config.json': train.sha_bytes(config), 'alignment.jsonl': train.sha_bytes(alignment)})
            manifest = dict(status='passed_tokenization_preflight', truncated_examples=0, files=files,
                            tokenizer=dict(actual_snapshot_commit=train.REVISION, eos_token_id=2,
                                           chat_template_sha256='a' * 64))
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            (directory / 'config.json').write_bytes(config)
            (directory / 'alignment.jsonl').write_bytes(alignment)
            # No train, calibration, development, or confirmation file exists.
            bundle = diagnostic.load_alignment_bundle(directory)
            self.assertEqual(len(bundle['rows']), 72)
            self.assertEqual(set(bundle['checked_prepared_files']), {'config.json', 'alignment.jsonl'})
            (directory / 'alignment.jsonl').write_bytes(alignment + b' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                diagnostic.load_alignment_bundle(directory)


if __name__ == '__main__':
    unittest.main()
