### Cloud GPU time (Hugging Face Jobs)

From the job records of the `lost-in-refusal-lexsi` account, read on 2026-11-25 15:10 UTC. Running time is what the record gives for each job; every job is counted, also the ones that failed or were stopped. Cost is running time at the list price of the GPU type, not the invoice.

| GPU type | Jobs | Completed | Running hours | List price an hour | Cost at list price |
|---|---:|---:|---:|---:|---:|
| `a100-large` | 1 | 1 | 0.35 | $2.50 | $0.88 |
| `l40sx1` | 13 | 12 | 6.17 | $1.80 | $11.11 |
| `l4x1` | 13 | 11 | 12.64 | $0.80 | $10.11 |
| all | 27 | | 19.16 | | $22.10 |

| Job | GPU type | Started (UTC) | End | Running minutes |
|---|---|---|---|---:|
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
