"""CPU checks for objective math, state restoration, and archive boundaries."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = importlib.import_module('experiments.multilingual_repair.train')

    def test_global_conflict_projection_and_rho_addition(self):
        # Dot=-2, ||a||²=5. Project (-6,2) to (-5.6,2.8), then add a.
        out, stats = self.api.combine_gradients([torch.tensor([-6.]), torch.tensor([2.])],
                                               [torch.tensor([1.]), torch.tensor([2.])], rho=1.)
        torch.testing.assert_close(torch.cat(out), torch.tensor([-4.6, 4.8]))
        self.assertTrue(stats['conflict'])
        self.assertAlmostEqual(stats['dot'], -2.)
        self.assertAlmostEqual(stats['projection_coefficient'], -.4)

    def test_locally_conflicting_tensor_is_not_projected_when_global_dot_positive(self):
        # A per-tensor projection would erroneously change the first coordinate.
        out, stats = self.api.combine_gradients([torch.tensor([-1.]), torch.tensor([3.])],
                                               [torch.tensor([1.]), torch.tensor([1.])], rho=1.)
        torch.testing.assert_close(torch.cat(out), torch.tensor([0., 4.]))
        self.assertFalse(stats['conflict'])

    def test_zero_alignment_at_initialization_is_valid(self):
        out, stats = self.api.combine_gradients([torch.tensor([2., -3.])], [torch.zeros(2)], rho=1.)
        torch.testing.assert_close(out[0], torch.tensor([2., -3.]))
        self.assertTrue(stats['zero_alignment'])
        self.assertEqual(stats['alignment_norm'], 0.)
        self.assertTrue(all(torch.isfinite(x).all() for x in out))

    def test_missing_nonfinite_and_wrong_shape_alignment_fail(self):
        for bad in (None, torch.tensor([float('nan')]), torch.tensor([float('inf')]), torch.ones(2)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.api.combine_gradients([torch.ones(1)], [bad], rho=1.)
        with self.assertRaises(ValueError):
            self.api.combine_gradients([torch.tensor([float('nan')])], [torch.ones(1)], rho=1.)

    def test_chunked_kl_matches_reference_to_student_batchmean_and_gradients(self):
        from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
        student = torch.tensor([[[1., 2., -1.], [0., 2., 1.], [3., 0., 1.]]], requires_grad=True)
        teacher = torch.tensor([[[3., 0., 1.], [1., 0., 2.], [-1., 2., 1.]]], requires_grad=True)
        loss = self.api.chunked_alignment_loss(student, teacher, safegrad_kl_alignment_loss, chunk_tokens=1)
        teacher_probs = torch.softmax(teacher.detach(), dim=-1)
        expected = (teacher_probs * (torch.log_softmax(teacher.detach(), dim=-1)
                                    - torch.log_softmax(student, dim=-1))).sum()
        torch.testing.assert_close(loss, expected)
        actual_grad = torch.autograd.grad(loss, student, retain_graph=True)[0]
        expected_grad = torch.softmax(student.detach(), dim=-1) - teacher_probs
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertIsNone(teacher.grad)

    @staticmethod
    def tiny_model():
        torch.manual_seed(4)
        base = LlamaForCausalLM(LlamaConfig(vocab_size=23, hidden_size=16, intermediate_size=24,
                                          num_hidden_layers=1, num_attention_heads=2,
                                          num_key_value_heads=2, max_position_embeddings=32,
                                          attention_dropout=0., use_cache=False))
        frozen = copy.deepcopy(base).eval()
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
                                               target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        return model, frozen

    def test_fixed_config_rejects_missing_or_sdpa_attention_backend(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'experiments/multilingual_repair/config.json').read_text())
        config['attention_implementation'] = 'eager'
        self.api.validate_config(config)
        for invalid in (None, 'sdpa'):
            modified = copy.deepcopy(config)
            if invalid is None:
                modified.pop('attention_implementation')
            else:
                modified['attention_implementation'] = invalid
            with self.assertRaisesRegex(ValueError, 'attention_implementation'):
                self.api.validate_config(modified)

    def test_initial_equivalence_rejects_nonzero_adapter_initialization(self):
        from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
        model, _ = self.tiny_model()
        with torch.no_grad():
            next(p for n, p in model.named_parameters() if '.lora_B.' in n).fill_(0.01)
        batch = {'input_ids': torch.tensor([[1, 2, 3, 4]]), 'attention_mask': torch.ones((1, 4), dtype=torch.long)}
        with self.assertRaisesRegex(ValueError, 'exactly zero LoRA B'):
            self.api.alignment_gradients(model, batch, [p for p in model.parameters() if p.requires_grad],
                                         safegrad_kl_alignment_loss, require_initial_equivalence=True)

    def test_eager_backend_is_pinned_and_effective_attention_configs_verified(self):
        model, _ = self.tiny_model()
        record = self.api.pin_attention_backend(model, 'eager')
        self.assertEqual(record['effective'], 'eager')
        self.assertEqual(model.config._attn_implementation, 'eager')
        self.assertEqual(record['attention_modules'], 1)
        model.config._attn_implementation = 'sdpa'
        with self.assertRaisesRegex(ValueError, 'attention backend'):
            self.api.verify_attention_backend(model, 'eager')
        with self.assertRaises(ValueError):
            self.api.pin_attention_backend(model, 'sdpa')

    def test_initial_equivalence_uses_grad_enabled_student_and_exact_zero_gradients(self):
        from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
        model, _ = self.tiny_model()
        self.api.pin_attention_backend(model, 'eager')
        batch = {'input_ids': torch.tensor([[1, 2, 3, 4]]), 'attention_mask': torch.ones((1, 4), dtype=torch.long)}
        parameters = [p for p in model.parameters() if p.requires_grad]
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        diagnostic = {}
        loss, gradients = self.api.alignment_gradients(model, batch, parameters, safegrad_kl_alignment_loss,
                                                       require_initial_equivalence=True, diagnostics=diagnostic)
        self.assertTrue(diagnostic['student_requires_grad'])
        self.assertTrue(diagnostic['initial_lora_B_zero'])
        self.assertTrue(diagnostic['reference_logits_equal'])
        self.assertEqual(diagnostic['max_abs_logit_difference'], 0.)
        self.assertTrue(diagnostic['initial_alignment_gradients_zero'])
        self.assertEqual(loss, 0.)
        self.assertTrue(all(torch.count_nonzero(g).item() == 0 for g in gradients))
        self.assertTrue(all(torch.equal(before[n], p.detach()) for n, p in model.named_parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_initial_equivalence_rejects_grad_only_execution_drift_before_mutation(self):
        from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
        model, _ = self.tiny_model()
        batch = {'input_ids': torch.tensor([[1, 2, 3, 4]]), 'attention_mask': torch.ones((1, 4), dtype=torch.long)}
        parameters = [p for p in model.parameters() if p.requires_grad]
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        flags = {n: p.requires_grad for n, p in model.named_parameters()}
        modes = {n: m.training for n, m in model.named_modules()}
        # Reproduce the observed backend failure: only the autograd path differs.
        def grad_only_drift(module, inputs, output):
            if torch.is_grad_enabled():
                delta = torch.zeros_like(output.logits)
                delta[:, :, 0] = 0.5
                output.logits = output.logits + delta
            return output
        hook = model.register_forward_hook(grad_only_drift)
        diagnostic = {}
        try:
            with self.assertRaisesRegex(ValueError, 'initial reference/student logits'):
                self.api.alignment_gradients(model, batch, parameters, safegrad_kl_alignment_loss,
                                             require_initial_equivalence=True, diagnostics=diagnostic)
        finally:
            hook.remove()
        self.assertTrue(diagnostic['student_requires_grad'])
        self.assertFalse(diagnostic['reference_logits_equal'])
        self.assertGreater(diagnostic['max_abs_logit_difference'], 0.)
        self.assertTrue(all(torch.equal(before[n], p.detach()) for n, p in model.named_parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(flags, {n: p.requires_grad for n, p in model.named_parameters()})
        self.assertEqual(modes, {n: m.training for n, m in model.named_modules()})

    def test_initial_equivalence_nonfinite_student_keeps_failure_diagnostics_serializable(self):
        from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
        for nonfinite in (float('nan'), float('inf')):
            with self.subTest(nonfinite=nonfinite):
                model, _ = self.tiny_model()
                batch = {'input_ids': torch.tensor([[1, 2, 3, 4]]),
                         'attention_mask': torch.ones((1, 4), dtype=torch.long)}
                parameters = [p for p in model.parameters() if p.requires_grad]
                flags = {n: p.requires_grad for n, p in model.named_parameters()}
                modes = {n: m.training for n, m in model.named_modules()}

                def grad_only_nonfinite(module, inputs, output):
                    if torch.is_grad_enabled():
                        output.logits = output.logits.clone()
                        output.logits[:, :, 0] = nonfinite
                    return output

                hook = model.register_forward_hook(grad_only_nonfinite)
                diagnostic = {}
                try:
                    with self.assertRaisesRegex(ValueError, 'Nonfinite alignment student logits'):
                        self.api.alignment_gradients(
                            model, batch, parameters, safegrad_kl_alignment_loss,
                            require_initial_equivalence=True, diagnostics=diagnostic)
                finally:
                    hook.remove()
                self.assertTrue(diagnostic['student_requires_grad'])
                self.assertNotIn('max_abs_logit_difference', diagnostic)
                self.assertNotIn('raw_pinned_helper_kl', diagnostic)
                json.dumps(diagnostic, allow_nan=False)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertEqual(flags, {n: p.requires_grad for n, p in model.named_parameters()})
                self.assertEqual(modes, {n: m.training for n, m in model.named_modules()})

    def test_reference_matches_original_base_and_restores_trainability_and_mode(self):
        model, frozen = self.tiny_model()
        ids = torch.tensor([[1, 2, 3, 4]])
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        original = {n: p.requires_grad for n, p in model.named_parameters()}
        modes = {n: m.training for n, m in model.named_modules()}
        with self.api.disabled_adapter_reference(model), torch.no_grad():
            self.assertFalse(model.training)
            self.assertFalse(model.is_gradient_checkpointing)
            observed = model(ids).logits
        with torch.no_grad():
            expected = frozen(ids).logits
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        self.assertEqual(original, {n: p.requires_grad for n, p in model.named_parameters()})
        self.assertEqual(modes, {n: m.training for n, m in model.named_modules()})
        self.assertTrue(model.is_gradient_checkpointing)
        model(ids, labels=ids).loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        with self.assertRaisesRegex(RuntimeError, 'abort'):
            with self.api.disabled_adapter_reference(model):
                raise RuntimeError('abort')
        self.assertEqual(original, {n: p.requires_grad for n, p in model.named_parameters()})
        self.assertEqual(modes, {n: m.training for n, m in model.named_modules()})

    def test_safegrad_tiny_model_zero_initial_adapter_then_nonzero_alignment(self):
        from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
        model, _ = self.tiny_model()
        batch = {'input_ids': torch.tensor([[1, 2, 3, 4]]), 'attention_mask': torch.ones((1, 4), dtype=torch.long)}
        params = [p for p in model.parameters() if p.requires_grad]
        loss, grads = self.api.alignment_gradients(model, batch, params, safegrad_kl_alignment_loss, chunk_tokens=2)
        self.assertAlmostEqual(loss, 0., places=5)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertEqual(loss, 0.)
        self.assertEqual(sum(g.norm().item() for g in grads), 0.)
        optimizer = torch.optim.SGD(params, lr=.2)
        model.train()
        model(**batch, labels=batch['input_ids']).loss.backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        loss, grads = self.api.alignment_gradients(model, batch, params, safegrad_kl_alignment_loss, chunk_tokens=2)
        self.assertGreater(loss, 0.)
        self.assertGreater(sum(g.norm().item() for g in grads), 0.)
        self.assertTrue(all(p.requires_grad for p in params))

    def test_selected_chunked_task_loss_matches_full_masked_causal_loss_and_gradients(self):
        model, _ = self.tiny_model()
        model.eval()
        row = dict(input_ids=[1, 2, 3, 4, 5], attention_mask=[1] * 5, labels=[-100, -100, -100, 4, 5],
                   prompt_tokens=3, sequence_tokens=5, supervised_tokens=2)
        expected = model(input_ids=torch.tensor([row['input_ids']]),
                         attention_mask=torch.tensor([row['attention_mask']]),
                         labels=torch.tensor([row['labels']])).loss
        actual = self.api.task_loss(model, row, device='cpu', chunk_tokens=1)
        torch.testing.assert_close(actual, expected)
        parameters = [p for p in model.parameters() if p.requires_grad]
        expected_grads = torch.autograd.grad(expected, parameters)
        actual_grads = torch.autograd.grad(actual, parameters)
        for expected_grad, actual_grad in zip(expected_grads, actual_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_training_order_never_accepts_calibration_and_feasibility_is_prefix(self):
        rows = [{'id': str(i), 'split': 'train'} for i in range(8)]
        whole = self.api.training_order(rows, seed=42)
        short = self.api.training_order(rows, seed=42, max_steps=3)
        self.assertEqual(short, whole[:3])
        self.assertEqual(set(whole), {str(i) for i in range(8)})
        with self.assertRaises(ValueError):
            self.api.training_order(rows + [{'id': 'c', 'split': 'calibration'}], seed=42)
        for steps in (0, 8, 9, -1):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.api.training_order(rows, seed=42, max_steps=steps)

    def test_token_validation_rejects_prompt_leak_missing_eos_and_truncation(self):
        row = dict(id='a', input_ids=[1, 3, 4, 2], attention_mask=[1, 1, 1, 1], labels=[-100, -100, 4, 2],
                   prompt_tokens=2, sequence_tokens=4, supervised_tokens=2)
        self.api.validate_token_row(row, max_tokens=4, eos_token_id=2)
        for changes in ({'labels': [1, -100, 4, 2]}, {'input_ids': [1, 3, 4, 5]}, {'sequence_tokens': 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.api.validate_token_row({**row, **changes}, max_tokens=4, eos_token_id=2)
        with self.assertRaises(ValueError):
            self.api.validate_token_row(row, max_tokens=3, eos_token_id=2)

    def test_prepared_hashes_and_safe_basenames_checked_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            # A corrupt arbitrary extra basename must also be checked, not ignored.
            (directory / 'extra.json').write_text('{}')
            (directory / 'manifest.json').write_text(json.dumps({'status': 'passed_tokenization_preflight', 'truncated_examples': 0,
                                                                                  'files': {'extra.json': '0' * 64}}))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.api.load_prepared(directory)
            (directory / 'manifest.json').write_text(json.dumps({'status': 'passed_tokenization_preflight', 'truncated_examples': 0,
                                                                                  'files': {'../escape.json': '0' * 64}}))
            with self.assertRaisesRegex(ValueError, 'safe basenames'):
                self.api.load_prepared(directory)


if __name__ == '__main__':
    unittest.main()
