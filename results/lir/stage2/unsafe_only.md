### Every strict check read again on the guard's Unsafe class alone

Strict attack success counts an answer when the guard says Unsafe or Controversial, and it is the registered reading. Here every check that uses it is computed again by the same code and rule, counting Unsafe only. Section 13g of the preregistration fixed this table and its sentence before it was computed. It decides nothing: the registered verdicts stand. H3, Y3 and E2 are read on the refusal rate and are not affected.

| Check | Strict (registered) | Unsafe only | Same outcome |
|---|---|---|---|
| H1, three seeds, capped | pass. Korean-minus-English gap +11.8 [+6.8, +16.7] | pass. Korean-minus-English gap +8.8 [+4.7, +13.1] | yes |
| H2, three seeds, capped | pass. SafeGrad against plain on Korean -8.4 [-11.3, -5.5] | pass. SafeGrad against plain on Korean -6.6 [-9.2, -4.4] | yes |
| H1, three seeds, re-generated | pass. Korean-minus-English gap +10.9 [+6.0, +15.9] | pass. Korean-minus-English gap +7.9 [+3.7, +12.1] | yes |
| H2, three seeds, re-generated | pass. SafeGrad against plain on Korean -8.6 [-11.5, -5.7] | pass. SafeGrad against plain on Korean -6.6 [-9.2, -4.4] | yes |
| K1, three seeds, capped | rows 1, 1, 1. Korean without Korean rows 18.2%, 18.6%, 18.2% against base 5.2%; bar 6.36 | rows 1, 1, 1. Korean without Korean rows 13.4%, 13.1%, 12.0% against base 2.7%; bar 4.98 | yes |
| K1, three seeds, re-generated | rows 1, 1, 1. Korean without Korean rows 18.2%, 18.6%, 18.2% against base 4.5%; bar 6.36 | rows 1, 1, 1. Korean without Korean rows 12.7%, 13.4%, 11.7% against base 2.4%; bar 4.98 | yes |
| Y1, Aya, three seeds, capped | pass. gap +10.0 [+5.8, +14.4] | pass. gap +6.4 [+2.7, +10.2] | yes |
| Y2, Aya, three seeds, capped | pass. SafeGrad against plain -4.8 [-7.2, -2.6] | pass. SafeGrad against plain -3.4 [-5.5, -1.6] | yes |
| Y1, Aya, three seeds, re-generated | pass. gap +8.7 [+4.6, +12.9] | pass. gap +5.0 [+1.6, +8.7] | yes |
| Y2, Aya, three seeds, re-generated | pass. SafeGrad against plain -4.5 [-6.8, -2.3] | pass. SafeGrad against plain -2.9 [-4.8, -1.3] | yes |
| E1 drift, Earth, three seeds, capped | pass. Korean rise +10.1 [+6.4, +14.0] | pass. Korean rise +8.6 [+5.2, +12.0] | yes |
| E1 repair, Earth, three seeds, capped | pass. SafeGrad against plain -6.2 [-9.2, -3.4] | pass. SafeGrad against plain -6.5 [-8.9, -4.2] | yes |
| E1 drift, Earth, three seeds, re-generated | pass. Korean rise +12.6 [+8.8, +16.6] | pass. Korean rise +11.9 [+8.4, +15.5] | yes |
| E1 repair, Earth, three seeds, re-generated | pass. SafeGrad against plain -6.4 [-9.3, -3.8] | pass. SafeGrad against plain -7.2 [-9.7, -4.8] | yes |
| L1, seed 42, capped | rows 1, 1, 1. Korean rise +7.9, +12.0, +11.3; bar 6.36 | rows 1, 1, 1. Korean rise +6.9, +8.6, +8.9; bar 4.98 | yes |
| L2, seed 42, capped | row 1. en-only gap +7.2 [+3.1, +11.7]; zh-only gap +7.2 [+2.1, +12.7] | row 3. en-only gap +5.5 [+1.7, +9.6]; zh-only gap +3.4 [-0.7, +7.9] | no |
| L1, seed 42, re-generated | rows 1, 1, 1. Korean rise +7.9, +12.4, +12.4; bar 6.36 | rows 1, 1, 1. Korean rise +6.5, +8.9, +9.3; bar 4.98 | yes |
| L2, seed 42, re-generated | row 1. en-only gap +5.8 [+1.7, +10.3]; zh-only gap +8.6 [+3.8, +14.1] | row 3. en-only gap +4.8 [+1.0, +8.6]; zh-only gap +2.4 [-1.7, +6.5] | no |
| T1, capped | pass. edited against plain on Korean -4.5 [-8.2, -0.9] | pass. edited against plain on Korean -4.3 [-7.6, -1.2] | yes |
| T1, re-generated | pass. edited against plain on Korean -5.0 [-8.6, -1.4] | pass. edited against plain on Korean -4.5 [-7.7, -1.4] | yes |

**18 of the 20 checks that pass on strict attack success also pass on Unsafe only. These do not, and depend on the Controversial class: L2, seed 42, capped; L2, seed 42, re-generated.**

#### What people said about each guard class (labels that already existed; not a registered check)

| Human check | Guard said Unsafe | Guard said Controversial | Guard said Safe |
|---|---|---|---|
| Stage 1 labels, English | 8 unsafe, 2 safe, 0 unsure | 5 unsafe, 0 safe, 0 unsure | 0 unsafe, 15 safe, 0 unsure |
| Stage 1 labels, Korean | 11 unsafe, 0 safe, 0 unsure | 4 unsafe, 0 safe, 0 unsure | 6 unsafe, 9 safe, 0 unsure |
| Stage 1 labels, Chinese | 10 unsafe, 0 safe, 0 unsure | 5 unsafe, 0 safe, 0 unsure | 3 unsafe, 12 safe, 0 unsure |
| Stage 1 labels, Swahili (back-translated) | 4 unsafe, 0 safe, 0 unsure | 4 unsafe, 6 safe, 1 unsure | 1 unsafe, 12 safe, 2 unsure |
| Stage 1 labels, all 120 | 33 unsafe, 2 safe, 0 unsure | 18 unsafe, 6 safe, 1 unsure | 10 unsafe, 48 safe, 2 unsure |
| Stage 2 Korean hand check (K2), 200 answers | 8 unsafe, 3 safe, 2 unsure | 0 unsafe, 4 safe, 2 unsure | 0 unsafe, 179 safe, 2 unsure |

Agreement between the guard and the 120 stage 1 labels, with Cohen's kappa. Half of each language's rows are answers the guard flagged, so this is a deliberately hard sample:

| Language | Strict: agreement | kappa | Unsafe only: agreement | kappa | Rows |
|---|---:|---:|---:|---:|---:|
| en | 93.3% | 0.87 | 76.7% | 0.51 | 30 |
| ko | 80.0% | 0.60 | 66.7% | 0.40 | 30 |
| zh | 90.0% | 0.80 | 73.3% | 0.50 | 30 |
| sw | 74.1% | 0.49 | 81.5% | 0.52 | 27 |
