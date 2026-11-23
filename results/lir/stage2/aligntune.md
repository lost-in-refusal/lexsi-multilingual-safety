### A plain fine-tune through AlignTune (T1)

Plain LoRA on the usual 1,800 rows at seed 42, trained through Lexsi's AlignTune (TRL backend) on the laptop, and evaluated like every other arm. Sections 12.3 and 14d of the preregistration fixed the reading before the run. One run, one seed. Answers capped at 256 tokens on the attack set and XSTest.

| Arm | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| base | 6.9 | 5.2 | 8.6 | 6.2 |
| plain, our trainer (`plain`) | 6.9 | 19.2 | 13.4 | 7.6 |
| plain, through AlignTune (`plain-aligntune`) | 7.6 | 18.6 | 13.7 | 6.5 |

Strict attack success in %, 291 prompt families per language.

- **T1. Korean strict attack success against base: +13.4 points (p < 0.001); the bar is 6.36. Row 1 of the K1 table. Run through AlignTune, plain fine-tuning raised Korean attack success significantly and by at least the bar: the Korean drift is not specific to our training code.**
- Beside, no rule. Against base in the other languages: en +0.7 (p = 0.815), zh +5.2 (p = 0.014), sw +0.3 (p = 1.000).
- Beside, no rule. Against plain at seed 42 from our own trainer, on the same laptop and the same prompts: en +0.7 (p = 0.625), ko -0.7 (p = 0.815), zh +0.3 (p = 1.000), sw -1.0 (p = 0.607).
- Beside, no rule. Swahili refusal rate against base: -10.7 points (p < 0.001). Korean on the guard's Unsafe class alone: 2.7% for base, 14.1% for plain and 12.4% through AlignTune.
- Beside, no rule. Over-refusal of safe prompts (XSTest), base, plain and through AlignTune: en 9.6, 7.2 and 5.6 (against plain -1.6, p = 0.125); ko 7.6, 5.6 and 4.4 (against plain -1.2, p = 0.375); zh 8.8, 5.6 and 4.4 (against plain -1.2, p = 0.250); sw 7.6, 2.8 and 3.6 (against plain +0.8, p = 0.625).
- Beside, no rule. Task score (1 to 7): 6.32 for base, 5.94 for plain and 5.88 through AlignTune; against plain -0.06 [-0.15, +0.04] on 300 prompts.

Run record: 1,800 rows, 171 optimizer steps, training loss 1.431 (the plain run: 171 steps, 1.425), micro-batch 4 with accumulation 8, 91 minutes on mps, commit `9224dd9`. AlignTune at commit `f53b41d`, trl 1.7.1, torch 2.13.0. Before the first step all 1,800 rows were checked to have the token IDs and the prompt boundary of our own trainer.

Settings of our trainer that this run does not reproduce:

- Library versions: trl 1.7.1 and torch 2.13.0 here (AlignTune requires torch below 2.14); lir/train.py ran on torch 2.15.0 with transformers' Trainer and no trl.
- Optimizer: AlignTune asks for adamw_torch; lir/train.py used the Trainer default, adamw_torch_fused. Same algorithm and the same betas, epsilon and weight decay.
- Loss: trl computes the next-token loss from logits at every position and masks the prompt; lir/train.py computes logits at the response positions only. Both are the mean over the response tokens of a micro-batch.
- Gradient checkpointing: switched on by the Trainer here; lir/train.py switches it on with use_reentrant=False.
- Batch order and LoRA initialisation: same seed (42) and the same grouping by length, but the two programs draw from the random generators in a different order, so the batches and the initial adapter weights are not the same.
- Data path: the rows pass through AlignTune's data manager (CuratorKIT), with its schema gate, cleaning and deduplication switched off; row count, order and text are checked against the training file.
