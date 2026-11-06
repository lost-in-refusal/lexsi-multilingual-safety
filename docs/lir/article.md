---
title: "Lost in Refusal: benign fine-tuning erodes Tiny Aya's safety unevenly across languages, and SafeGrad repairs it"
authors: [aishwarya-nikkumbh, xtnayk, pham215]
---

<!-- Draft for Aish to own and edit. Numbers marked {{...}} are filled from results/lir/results.json. -->

# Lost in Refusal

**TL;DR.** {{one-sentence headline result}}

## 1. Why this matters

Small multilingual models are what teams actually fine-tune and self-host: they fit a single GPU and cost a
fraction of a closed API ({{ROI line}}). Fine-tuning on harmless data is known to erode safety training (Qi et al.,
2023). Tiny Aya's own report notes that its safety in a language depends on having safety data in that language,
and that preference tuning was kept light so the community could adapt the model. So the exact users Tiny Aya was
built for, multilingual teams adapting it, are the ones most exposed to drift, and they will not have safety data
in every language.

**Use case.** {{named deployment, e.g. a multilingual study/support assistant serving English, Korean and Mandarin
users, with Swahili on the roadmap}}

## 2. Setup

- Model: Tiny Aya Global (3.35B), default chat template, greedy decoding.
- Fine-tuning data: 600 Bactrian-X instructions × en/ko/zh (1,800 rows), cleaned and deduplicated with CuratorKIT,
  screened with Qwen3Guard. Benign only.
- Arms: plain LoRA vs SafeTune SafeGrad LoRA (identical data, order, LoRA and optimizer).
- Attacks: 291 MultiJail families in en/ko/zh and Swahili, which never appears in fine-tuning.
- Over-refusal: XSTest's 250 safe prompts in the same four languages.
- Task: 100 held-out Bactrian-X instructions per training language, Qwen3-14B judge, Tiny Aya's 1–7 rubric.
- Audit: AuditKIT, paired bootstrap against the base model.

## 3. Break: benign fine-tuning removes refusals, unevenly

{{table: attack success per language, base vs plain}}

{{finding: which language drifts most; Swahili transfer}}

## 4. Fix: SafeGrad keeps refusal without giving up the task

{{table: three arms × ASR / over-refusal / task score}}

{{surgery stats: % of steps where the task and alignment gradients conflicted}}

## 5. Explain: where refusal lives, and why other languages slip past

{{figure: results/lir/refusal_geometry.png}}

{{probe findings: cosine of per-language refusal directions to English; English-direction AUROC per language;
linear vs MLP probe transfer; projection drift after fine-tuning}}

{{CircuitKIT: cross-language overlap of refusal circuits}}

{{steering: English direction vs native direction, ASR restored per language}}

## 6. ROI

{{ROI table: cost per 1M tokens self-hosted vs closed API; SafeGrad training overhead}}

## 7. Feedback to Lexsi

See `docs/lir/lexsi-feedback.md` and the `lexsi-feedback` issues.

## 8. Limitations

- Automatic judges; human agreement checked on {{n}} items per language ({{agreement}}).
- Single seed, one LoRA configuration, one model size.
- Machine-translated XSTest; Swahili judged without a fluent reader (back-translation check only).

## Reproduce

Everything is public under the `lost-in-refusal` Hugging Face org; the notebook recomputes every number from
the published outputs on CPU.
