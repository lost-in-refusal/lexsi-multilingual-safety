"""CPU-only Cohere regression for the frozen reference and checkpointed KL."""
import copy
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import CohereConfig, CohereForCausalLM

from experiments.multilingual_repair import train
from safetune.core.optim.safegrad import safegrad_kl_alignment_loss


class CohereReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def setUp(self):
        # Construct random CPU weights directly: no pretrained model or device probe.
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.random.default_generator.manual_seed(123)
            base = CohereForCausalLM(CohereConfig(
                vocab_size=29, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=32, attention_dropout=.4,
                logit_scale=.2, use_cache=False, bos_token_id=1, eos_token_id=2,
                pad_token_id=0,
            )).to(dtype=torch.bfloat16)
            self.frozen = copy.deepcopy(base).eval().requires_grad_(False)
            self.model = get_peft_model(base, LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.,
                target_modules=sorted(train.TARGETS), task_type='CAUSAL_LM',
            ))
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if 'lora_B' in name:
                        parameter.normal_(std=.1)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]], device='cpu'),
            'attention_mask': torch.ones((1, 5), dtype=torch.long, device='cpu'),
        }
        self.assertTrue(all(p.device.type == 'cpu' for p in self.model.parameters()))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in self.frozen.parameters()))
        self.assertTrue(all(p.dtype == torch.float32 for p in self.parameters))
        self.assertTrue(all(torch.count_nonzero(p).item() > 0
                            for n, p in self.model.named_parameters() if 'lora_B' in n))
        self.model.train()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        # Preserve mixed child-module modes as well as the top-level train flag.
        self.model.get_base_model().model.layers[0].self_attn.eval()
        self.flags = {n: p.requires_grad for n, p in self.model.named_parameters()}
        self.modes = {n: m.training for n, m in self.model.named_modules()}

    def assert_state_restored(self):
        self.assertEqual(self.flags, {n: p.requires_grad for n, p in self.model.named_parameters()})
        self.assertEqual(self.modes, {n: m.training for n, m in self.model.named_modules()})
        self.assertTrue(self.model.is_gradient_checkpointing)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_nonzero_adapter_reference_and_checkpointed_gradients_match_dense(self):
        with train.disabled_adapter_reference(self.model), torch.no_grad():
            self.assertTrue(all(not m.training for m in self.model.modules()))
            self.assertFalse(self.model.is_gradient_checkpointing)
            reference = self.model(**self.batch).logits
        with torch.no_grad():
            original = self.frozen(**self.batch).logits
        torch.testing.assert_close(reference, original, rtol=0, atol=0)
        self.assert_state_restored()

        actual_loss, actual_gradients = train.alignment_gradients(
            self.model, self.batch, self.parameters, safegrad_kl_alignment_loss,
            chunk_tokens=2,
        )
        self.assert_state_restored()
        with train.model_mode(self.model, training=False, checkpointing=False):
            student = self.model(**self.batch).logits
            self.assertFalse(torch.equal(student.detach(), reference))
            expected_loss = safegrad_kl_alignment_loss(
                student.float(), reference.float(), temperature=1., reduction='batchmean')
            expected_gradients = torch.autograd.grad(expected_loss, self.parameters)
        self.assertGreater(actual_loss, 0.)
        self.assertEqual(actual_loss, expected_loss.item())
        self.assertEqual(len(actual_gradients), len(self.parameters))
        self.assertGreater(sum(g.norm().item() for g in actual_gradients), 0.)
        for actual, expected in zip(actual_gradients, expected_gradients):
            self.assertTrue(torch.isfinite(actual).all().item())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_state_restored()

        with self.assertRaisesRegex(RuntimeError, 'abort reference'):
            with train.disabled_adapter_reference(self.model):
                raise RuntimeError('abort reference')
        self.assert_state_restored()

    def test_bfloat16_cohere_task_shift_matches_builtin_masked_loss(self):
        row = dict(input_ids=[1, 2, 3, 4, 5], attention_mask=[1] * 5,
                   labels=[-100, -100, -100, 4, 5], prompt_tokens=3,
                   sequence_tokens=5, supervised_tokens=2)
        with train.model_mode(self.model, training=False, checkpointing=False):
            expected = self.model(
                **self.batch, labels=torch.tensor([row['labels']], device='cpu')).loss
            actual = train.task_loss(self.model, row, device='cpu', chunk_tokens=1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_state_restored()


if __name__ == '__main__':
    unittest.main()
