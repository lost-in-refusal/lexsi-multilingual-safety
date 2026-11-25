### The Korean rise on the summarisation fine-tune, by answer length and the cap (section 20b: C1)

The Korean attack answers of `xlsum-base` and of the plain runs at seeds 42, 1 and 2, split as in section 18c: by whether the answer reached the 256-token cap, and into three groups of equal size by its length in tokens. Inside a group: strict attack success of the plain answers, the seeds put together, minus that of the base answers, with a 95% interval from a bootstrap over the prompt families.

| Group | Tokens | Of them at the cap | Base answers | flagged (%) | Plain answers | flagged (%) | Plain minus base (points) | Interval |
|---|---|---:|---:|---:|---:|---:|---|---|
| Complete answers | 8 to 255 | 0 | 194 | 0.5 | 688 | 6.8 | +6.3 [+3.8, +9.1] | above zero |
| Answers that reached the cap | 256 to 256 | 282 | 97 | 13.4 | 185 | 18.4 | +5.0 [-2.7, +13.1] | includes zero |
| Shortest third | 8 to 58 | 0 | 42 | 2.4 | 346 | 6.4 | +4.0 [-0.6, +8.3] | includes zero |
| Middle third | 58 to 200 | 0 | 118 | 0.0 | 270 | 6.7 | +6.7 [+3.3, +10.6] | above zero |
| Longest third | 201 to 256 | 282 | 131 | 9.9 | 257 | 16.0 | +6.0 [-0.1, +12.4] | includes zero |

**The Korean rise is not found in the shortest third and the longest third, so it cannot be separated from the change in answer length there.**

The thirds are cut over the 1,164 answers together (291 of the base run, counted once). A group is defined by the answer, which the fine-tune changes, so this compares different prompts in the two runs: a description, not a control. No answer of this block was generated again past the cap.
