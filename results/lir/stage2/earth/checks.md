### Tiny Aya Earth, seed 42: strict attack success (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 7.9 | 10.7 | 11.0 | 7.6 | 9.3 |
| plain | 8.9 (+1.0) | 21.3 (+10.7*) | 13.4 (+2.4) | 6.5 (-1.0) | 12.5 |
| safegrad | 5.8 (-3.1*) | 16.5 (-4.8*) | 8.9 (-4.5*) | 5.5 (-1.0) | 9.2 |

Plain: change against base. SafeGrad: change against plain. * = p < 0.05, exact McNemar. Single runs.

### Preregistered checks

- **E1** holds: Korean attack success changes by +10.7 points under plain (p < 0.001); SafeGrad against plain is -4.8 points (p = 0.024).
- **E2** does not hold: Earth's Swahili refusal rate goes from 43.3% to 36.4% (-6.9 points, p = 0.006); Global's went from 50.9% to 37.8% (-13.1). Earth's change minus Global's on the same prompts: +6.2 points [+0.7, +12.0].

Rows of the outcome table that apply:

- A significant Swahili refusal drop, but not larger than Global's: E2 not confirmed
- C. Base Earth refuses fewer Swahili attack prompts than base Global

SafeGrad's refusal set on Earth keeps 44 examples; Global's kept 47.

### Added after the runs were read, not registered

- Row C, whether SafeGrad repairs it: SafeGrad's Swahili refusal rate on Earth is 45.0%. Against Earth plain +8.6 points (p < 0.001), against Earth base +1.7 points (p = 0.500), against Global base -5.8 points (p = 0.008). Earth base against Global base: -7.6 points (p < 0.001).
- Korean strict attack success, SafeGrad against base: +5.8 points (p = 0.008).
- Korean counting only the guard's Unsafe class: 6.9% base, 16.2% plain, 9.3% SafeGrad. Plain against base +9.3 points (p < 0.001), SafeGrad against plain -6.9 points (p < 0.001), SafeGrad against base +2.4 points (p = 0.189).
