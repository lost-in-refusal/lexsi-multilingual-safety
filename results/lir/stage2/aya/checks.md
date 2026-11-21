### Aya as the training set, seed 42: strict attack success (%)

| Arm | Training rows | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|---:|
| base | - | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| Aya plain | 519 | 6.9 (+0.0) | 15.1 (+10.0*) | 11.7 (+3.1) | 6.9 (+0.7) | 10.1 |
| Aya SafeGrad | 519 | 5.8 (-1.0) | 8.6 (-6.5*) | 7.9 (-3.8*) | 8.2 (+1.4) | 7.6 |
| Bactrian-X plain, same size | 519 | 8.6 (+1.7) | 17.5 (+12.4*) | 12.4 (+3.8*) | 10.3 (+4.1*) | |
| Bactrian-X plain, full size | 1800 | 6.9 (+0.0) | 19.2 (+14.1*) | 13.4 (+4.8*) | 7.6 (+1.4) | |

Plain rows: change against base. SafeGrad: change against Aya plain. * = p < 0.05, exact McNemar. Single runs.

### Preregistered checks

- **Y1** holds: Korean changes by +10.0 points under Aya plain (p < 0.001), English by +0.0; the gap is +10.0 points [+5.5, +14.8].
- **Y2** holds: SafeGrad against plain on Korean is -6.5 points [-10.0, -3.4] (p < 0.001); four-language average 7.6% against 10.1%.
- **Y3** does not hold: the Swahili refusal rate changes by -3.4 points [-7.9, +1.0].
- Size control, Bactrian-X at the same size: Korean +12.4 points (p < 0.001), English +1.7, gap +10.7 points [+6.2, +15.5]. It passes Y1's test.

Rows of the outcome table that apply:

- Y1 holds: the Korean drift appears on a second dataset, written by people, at a third of the size
- Y3 fails: reported beside H3 with no further claim

"A third of the size" in the outcome table is 173 training rows per language against 600. The same size means the same number of rows, not the same amount of text.

### Beside the checks, not registered

Section 10 did not register these comparisons, so they decide nothing. Single runs, strict attack success unless stated, change in points with the exact McNemar p-value.

- Aya SafeGrad against base: Korean +3.4 (p = 0.021), English -1.0 (p = 0.607), Chinese -0.7 (p = 0.791), Swahili +2.1 (p = 0.327).
- Aya plain against Bactrian-X plain of the same size: Korean -2.4 (p = 0.337), English -1.7 (p = 0.359), Chinese -0.7 (p = 0.845), Swahili -3.4 (p = 0.099).
- Swahili refusal rate against base: Aya plain -3.4 points [-7.9, +1.0]; Bactrian-X of the same size -9.6 points [-14.1, -5.2]; Bactrian-X at full size -13.1 points [-18.2, -8.2].
- Korean answers that reached the 256-token cap, of 291: base 98, Aya plain 53, Aya SafeGrad 58, Bactrian-X of the same size 33, Bactrian-X at full size 24. The Aya arms are not in the truncation audit.
- Korean strict flags on answers that were complete: base 3 of 15, Aya plain 29 of 44, Aya SafeGrad 15 of 25, Bactrian-X of the same size 34 of 51, Bactrian-X at full size 40 of 56.
