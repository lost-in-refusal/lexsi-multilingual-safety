# ROI: self-hosting a hardened Tiny Aya support bot vs a closed API

Scenario: a customer-support bot for English, Korean and Mandarin customers, self-hosted on Tiny Aya Global and
fine-tuned on in-house data, against calling a closed model per request.

Numbers measured in our runs are marked **(measured)**; prices are inputs dated 2026-11-06 (OpenAI list prices as
reported for Oct 2026; confirm on openai.com/api/pricing before quoting). Source: `results/lir/roi.json`
(`python -m lir.roi`).

## One evaluation pass (2,463 requests: 1,164 attacks, 999 XSTest, 300 tasks)

| | Tokens | Cost / time |
|---|---:|---:|
| Input tokens (incl. Tiny Aya's default system preamble) **(measured)** | 935k | |
| Output tokens, base model **(measured)** | 453k | |
| GPT-5 mini at $0.25 / $2.00 per 1M in/out | | **$1.18** |
| GPT-5 nano at $0.05 / $0.40 per 1M in/out | | **$0.24** |
| Tiny Aya Global on one Apple M5 Pro laptop, plain HF transformers, bf16 **(measured)** | 140–280 output tok/s | **0.57 h** |

**Break-even.** Self-hosting on this laptop costs less than GPT-5 mini whenever the machine costs under
**$2.08 per hour** to own and run, and less than GPT-5 nano under **$0.42 per hour**. A laptop in this class
amortised over three years of 8-hour days, plus power, is well under the first figure and around the second. Our
inference is unoptimised (Hugging Face `generate`, no batching server), so these are conservative for
self-hosting; a GPU server with vLLM would be several times faster, but we did not measure that.

## The safety fix

| | Plain LoRA | SafeGrad LoRA | Difference |
|---|---:|---:|---:|
| Training time, 1,800 rows × 3 epochs **(measured)** | 75.7 min | 89.7 min | +14.0 min (+18.5%) |
| Final training loss **(measured)** | 1.425 | 1.436 | +0.8% |
| Korean attack success, strict **(measured)** | 19.2% | 10.7% | −8.6 pts |
| Mandarin attack success, strict **(measured)** | 13.4% | 8.9% | −4.5 pts |
| Refusal on Swahili attacks **(measured)** | 37.8% | 56.4% | +18.6 pts |
| Over-refusal on XSTest, macro **(measured)** | 5.3% | 6.4% | +1.1 pts |

The hardening costs 14 minutes of laptop time per fine-tune. Skipping it nearly quadruples Korean attack success
relative to the base model, and an English-only safety check would not notice.
