"""Check miniature Cohere architectures on Metal without downloading weights.

This verifies local API/kernel wiring, not the actual Aya checkpoints or their
memory/performance, and does not validate SafeGrad's alignment surgery.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, CohereConfig, Cohere2Config
from safetune._refusal_helpers import _get_decoder_layers
from safetune.runner.utils.model_utils import lora_wrap


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    results = []
    common = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    configs = [
        CohereConfig(**common),
        Cohere2Config(**common, sliding_window=64,
                      layer_types=["sliding_attention", "full_attention"]),
    ]
    for config in configs:
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config).to(device="mps", dtype=torch.bfloat16)
        assert len(_get_decoder_layers(model)) == 2
        model = lora_wrap(model, r=4, lora_alpha=8)
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in trainable]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        tokens = torch.tensor([[1, 5, 6, 7, 8, 2]], device="mps")
        loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
        assert torch.isfinite(loss).item()
        loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all().item() for g in gradients)
        optimizer.step()
        torch.mps.synchronize()
        assert any(not torch.equal(old, p) for old, p in zip(before, trainable))
        results.append({"architecture": config.model_type, "device": "mps",
                        "loss": loss.item(), "lora_optimizer_step": "passed"})
        del model, optimizer, trainable, before, gradients, loss
        torch.mps.empty_cache()
    report = {"scope": "Miniature random models only; no Aya checkpoint or SafeGrad validation.",
              "results": results}
    Path("results").mkdir(exist_ok=True)
    Path("results/mps-architecture-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
