### Cloud GPU time (Hugging Face Jobs)

From the job records of the `lost-in-refusal-lexsi` account, read on 2026-11-25 23:53 UTC. Running time is what the record gives for each job; every job is counted, also the ones that failed or were stopped. Cost is running time at the list price of the GPU type, not the invoice.

| GPU type | Jobs | Completed | Running hours | List price an hour | Cost at list price |
|---|---:|---:|---:|---:|---:|
| `a100-large` | 15 | 15 | 5.89 | $2.50 | $14.72 |
| `l40sx1` | 14 | 12 | 6.17 | $1.80 | $11.11 |
| `l4x1` | 13 | 11 | 12.64 | $0.80 | $10.11 |
| all | 42 | | 24.70 | | $35.94 |

| Job | GPU type | Started (UTC) | End | Running minutes |
|---|---|---|---|---:|
| `lir-xlsum-build` | `l40sx1` |  | canceled | 0.0 |
| `lir-control` | `l4x1` | 2026-11-22 20:56 | completed | 65.6 |
| `lir-only-ko` | `l4x1` | 2026-11-22 21:28 | completed | 50.5 |
| `lir-nf4` | `l4x1` | 2026-11-22 21:49 | completed | 136.9 |
| `lir-only-zh` | `l4x1` | 2026-11-22 22:25 | completed | 50.7 |
| `lir-only-en` | `l4x1` | 2026-11-22 22:25 | completed | 51.7 |
| `lir-d13` | `l4x1` | 2026-11-23 04:19 | error | 3.0 |
| `lir-d13` | `l4x1` | 2026-11-23 05:01 | error | 2.8 |
| `lir-unmerged-s2` | `l4x1` | 2026-11-25 01:05 | completed | 71.2 |
| `lir-nf4-s1` | `l4x1` | 2026-11-25 01:05 | completed | 73.0 |
| `lir-nf4-s2` | `l4x1` | 2026-11-25 01:17 | completed | 73.7 |
| `lir-unmerged` | `l4x1` | 2026-11-25 01:34 | completed | 71.6 |
| `lir-unmerged-s1` | `l4x1` | 2026-11-25 01:46 | completed | 72.2 |
| `lir-mmlu` | `l4x1` | 2026-11-25 02:22 | completed | 35.4 |
| `lir-granite41-base` | `l40sx1` | 2026-11-25 02:22 | completed | 10.2 |
| `lir-granite41-plain-s2` | `l40sx1` | 2026-11-25 02:32 | completed | 24.3 |
| `lir-granite41-plain-s42` | `l40sx1` | 2026-11-25 02:32 | completed | 23.1 |
| `lir-granite41-plain-s1` | `l40sx1` | 2026-11-25 02:44 | completed | 23.2 |
| `lir-granite41-safegrad-s42` | `l40sx1` | 2026-11-25 02:47 | completed | 25.4 |
| `lir-granite41-safegrad-s1` | `l40sx1` | 2026-11-25 03:13 | completed | 25.1 |
| `lir-granite41-safegrad-s2` | `l40sx1` | 2026-11-25 03:17 | completed | 26.3 |
| `lir-granite41-regen` | `l40sx1` | 2026-11-25 04:25 | completed | 65.5 |
| `lir-granite41-regen` | `l40sx1` | 2026-11-25 05:05 | error | 22.4 |
| `lir-granite41-noko-s2` | `l40sx1` | 2026-11-25 05:36 | completed | 21.9 |
| `lir-granite41-noko-s42` | `l40sx1` | 2026-11-25 05:39 | completed | 22.4 |
| `lir-granite41-noko-s1` | `l40sx1` | 2026-11-25 05:50 | completed | 21.7 |
| `lir-granite41-task` | `l40sx1` | 2026-11-25 05:57 | completed | 58.6 |
| `lir-judge3-labels` | `a100-large` | 2026-11-25 12:37 | completed | 21.1 |
| `lir-xlsum-build` | `a100-large` | 2026-11-25 21:18 | completed | 5.2 |
| `lir-xlsum-base` | `a100-large` | 2026-11-25 21:26 | completed | 17.5 |
| `lir-xlsum-plain-s42` | `a100-large` | 2026-11-25 21:29 | completed | 30.3 |
| `lir-xlsum-judge` | `a100-large` | 2026-11-25 22:01 | completed | 6.4 |
| `lir-xlsum-safegrad-s2` | `a100-large` | 2026-11-25 22:10 | completed | 32.3 |
| `lir-xlsum-safegrad-s1` | `a100-large` | 2026-11-25 22:10 | completed | 32.5 |
| `lir-xlsum-safegrad-s42` | `a100-large` | 2026-11-25 22:10 | completed | 31.7 |
| `lir-xlsum-plain-s2` | `a100-large` | 2026-11-25 22:10 | completed | 30.6 |
| `lir-xlsum-plain-s1` | `a100-large` | 2026-11-25 22:10 | completed | 29.7 |
| `lir-xlsum-judge` | `a100-large` | 2026-11-25 22:43 | completed | 13.0 |
| `lir-xlsum-mix5-s2` | `a100-large` | 2026-11-25 23:11 | completed | 31.6 |
| `lir-xlsum-mix5-s1` | `a100-large` | 2026-11-25 23:11 | completed | 31.6 |
| `lir-xlsum-mix5-s42` | `a100-large` | 2026-11-25 23:11 | completed | 31.2 |
| `lir-xlsum-judge` | `a100-large` | 2026-11-25 23:43 | completed | 8.6 |
