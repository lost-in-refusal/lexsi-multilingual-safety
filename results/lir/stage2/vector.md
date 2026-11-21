### Repair vector added to another seed's plain model: strict attack success (%)

Exploratory and not preregistered. Single runs. The vector is SafeGrad minus plain at seed 42; no training and no tuning on these prompts.

| Seed | Arm | en | ko | zh | sw | macro |
|---|---|---:|---:|---:|---:|---:|
| 1 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 1 | plain | 7.6 | 15.5 | 12.4 | 5.2 | 10.1 |
| 1 | safegrad | 3.4 | 8.2 | 7.6 | 3.4 | 5.7 |
| 1 | plain + vector | 5.2 (-2.4) | 10.3 (-5.2*) | 9.6 (-2.7) | 5.2 (+0.0) | 7.6 |
| 2 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 2 | plain | 8.9 | 18.9 | 13.1 | 6.9 | 11.9 |
| 2 | safegrad | 4.1 | 9.6 | 8.9 | 6.5 | 7.3 |
| 2 | plain + vector | 5.2 (-3.8*) | 11.0 (-7.9*) | 10.7 (-2.4) | 5.8 (-1.0) | 8.2 |

Parentheses: change against the plain model of the same seed. * = p < 0.05, exact McNemar.

Seed 1, Korean: the edit gives 71% of the drop that training with SafeGrad gave; edited against SafeGrad +2.1 points (p = 0.146).
Seed 2, Korean: the edit gives 85% of the drop that training with SafeGrad gave; edited against SafeGrad +1.4 points (p = 0.388).

Over-refusal and task score for the edited models are in `vector/results.md`. They are exploratory too.
