### Global-MMLU-Lite on the arms of section 15 (section 15b: B1)

`CohereLabs/Global-MMLU-Lite` at revision `36c2fd75`: the 400 test questions of each of English, Korean, Chinese and Swahili, four options each. Each arm is built as in section 15, the plan applied on the CPU and written out as plain weights, and run on one GPU in a Hugging Face job; the laptop's arms are not these processes. The answer is the option letter with the highest probability for the next token after the question, with one instruction in English. No judge model reads anything. Chance is 25%. A difference averages a question over its languages, then bootstraps the questions (95% interval).

Job upload `20261125-022435-01c70c1`, commit `01c70c1`, NVIDIA L4, 620.70.06, 23034 MiB.

**B1. The circuit plan scored below uniform 4-bit on Global-MMLU-Lite.** Circuit plan minus uniform 4-bit, accuracy over the four languages: -14.1 [-18.2, -10.1] points.

Accuracy (%), with no rule:

| Arm | en | ko | zh | sw | Four languages | Against the unquantized model |
|---|---|---|---|---|---|---|
| `bf16` | 59.0 | 55.5 | 54.8 | 48.8 | 54.5 |  |
| `int4` | 58.2 | 51.5 | 50.0 | 45.0 | 51.2 | -3.3 [-5.5, -1.2] |
| `circuit` | 41.0 | 37.5 | 38.5 | 31.2 | 37.1 | -17.4 [-21.5, -13.4] |
| `random-0` | 42.2 | 31.2 | 29.5 | 32.0 | 33.8 | -20.8 [-25.3, -16.2] |
| `random-2` | 49.8 | 41.2 | 41.8 | 38.0 | 42.7 | -11.8 [-15.6, -7.9] |
| `random-4` | 49.8 | 43.8 | 45.5 | 41.8 | 45.2 | -9.3 [-12.6, -5.9] |

The circuit plan minus each random plan, with no rule: `random-0` +3.3 [-0.7, +7.5]; `random-2` -5.6 [-9.6, -1.8]; `random-4` -8.1 [-12.1, -4.0].

Probability on the four option letters, mean over the questions: `bf16` 0.98, `circuit` 0.76, `int4` 0.98, `random-0` 0.61, `random-2` 0.92, `random-4` 0.84. A low figure would mean the model wants to write something other than a letter, and the score would then say less.

This does not enter Q0 to Q3 or the table of section 15. Not measured: the fine-tuned models, IFEval, and any question outside these four languages.
