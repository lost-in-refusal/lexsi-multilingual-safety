Answers that reached the 256-token cap were generated again with a cap of 1024 tokens and judged again (preregistration section 8). The table on the capped answers is `harm_categories.md`.

### Strict attack success by harm category, averaged over seeds 1, 2, 42

A prompt with several tags counts in each of its categories, so rows overlap. Changes are in points with a 95% interval over the category's prompts. Loose attack success is in brackets after each rate.

Korean strict attack success rises under plain in every category with 20 or more prompts.

#### Korean

| Category | Prompts | Base | Plain | SafeGrad | Plain against base | SafeGrad against plain |
|---|---:|---:|---:|---:|---|---|
| Violence & incitement | 80 | 6.2 (5.0) | 14.2 (11.2) | 8.8 (6.2) | +7.9 [+3.3, +13.3] | -5.4 [-10.4, -0.8] |
| Discrimination & injustice | 46 | 2.2 (2.2) | 23.2 (12.3) | 12.3 (8.0) | +21.0 [+10.9, +31.9] | -10.9 [-18.8, -4.3] |
| Hate speech & offensive language | 36 | 8.3 (5.6) | 27.8 (21.3) | 25.0 (18.5) | +19.4 [+6.5, +33.3] | -2.8 [-9.3, +4.6] |
| Bullying & harassment | 29 | 3.4 (0.0) | 20.7 (11.5) | 11.5 (4.6) | +17.2 [+3.4, +32.2] | -9.2 [-17.2, -3.4] |
| Non-violent unethical behavior (e.g. lying, cheating, etc.) | 29 | 3.4 (0.0) | 21.8 (17.2) | 10.3 (8.0) | +18.4 [+3.4, +34.5] | -11.5 [-23.0, -2.3] |
| Theft | 27 | 3.7 (0.0) | 7.4 (4.9) | 3.7 (0.0) | +3.7 [-7.4, +14.8] | -3.7 [-9.9, +0.0] |
| Conspiracy theories & misinformation | 26 | 0.0 (0.0) | 12.8 (12.8) | 3.8 (3.8) | +12.8 [+2.6, +25.6] | -9.0 [-19.2, -1.3] |
| Weapons | 21 | 9.5 (9.5) | 30.2 (28.6) | 14.3 (14.3) | +20.6 [+6.3, +36.5] | -15.9 [-30.2, -4.8] |
| *Fewer than 20 prompts: listed, not interpreted* | | | | | | |
| Adult content | 19 | 5.3 (0.0) | 24.6 (14.0) | 14.0 (5.3) | +19.3 [+3.5, +38.6] | -10.5 [-26.3, +1.8] |
| Fraud & deception | 17 | 5.9 (0.0) | 29.4 (25.5) | 15.7 (9.8) | +23.5 [+7.8, +43.1] | -13.7 [-27.5, -3.9] |
| Substance abuse & banned substances | 16 | 6.2 (6.2) | 16.7 (10.4) | 2.1 (2.1) | +10.4 [-4.2, +29.2] | -14.6 [-33.3, +0.0] |
| Terrorism & organized crime | 16 | 6.2 (6.2) | 8.3 (4.2) | 2.1 (0.0) | +2.1 [-10.4, +14.6] | -6.2 [-12.5, +0.0] |
| Animal abuse | 11 | 0.0 (0.0) | 6.1 (3.0) | 0.0 (0.0) | +6.1 [+0.0, +18.2] | -6.1 [-18.2, +0.0] |
| Sexual exploitation & human trafficking | 11 | 18.2 (9.1) | 36.4 (27.3) | 18.2 (15.2) | +18.2 [+0.0, +45.5] | -18.2 [-45.5, +0.0] |
| Property crime & vandalism | 10 | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Soliciting personally identifiable information | 10 | 0.0 (0.0) | 6.7 (6.7) | 10.0 (10.0) | +6.7 [+0.0, +20.0] | +3.3 [+0.0, +10.0] |
| Self-harm | 8 | 0.0 (0.0) | 45.8 (29.2) | 0.0 (0.0) | +45.8 [+12.5, +79.2] | -45.8 [-79.2, -12.5] |
| Child abuse | 6 | 16.7 (0.0) | 27.8 (5.6) | 33.3 (11.1) | +11.1 [+0.0, +33.3] | +5.6 [+0.0, +16.7] |

#### Four languages together

| Category | Prompts | Base | Plain | SafeGrad | Plain against base | SafeGrad against plain |
|---|---:|---:|---:|---:|---|---|
| Violence & incitement | 80 | 4.4 (3.1) | 10.9 (7.7) | 6.0 (4.0) | +6.6 [+3.9, +9.7] | -4.9 [-8.1, -2.2] |
| Discrimination & injustice | 46 | 4.3 (1.1) | 9.6 (4.5) | 5.8 (2.7) | +5.3 [+0.4, +9.6] | -3.8 [-6.3, -1.4] |
| Hate speech & offensive language | 36 | 11.1 (3.5) | 15.3 (10.6) | 13.0 (8.6) | +4.2 [-1.4, +9.3] | -2.3 [-5.1, +0.0] |
| Bullying & harassment | 29 | 0.9 (0.0) | 8.9 (5.5) | 4.0 (1.4) | +8.0 [+2.6, +14.4] | -4.9 [-9.5, -0.9] |
| Non-violent unethical behavior (e.g. lying, cheating, etc.) | 29 | 6.0 (2.6) | 11.5 (6.9) | 6.6 (3.7) | +5.5 [+0.9, +10.9] | -4.9 [-10.1, -0.3] |
| Theft | 27 | 3.7 (0.9) | 4.6 (2.2) | 2.8 (0.9) | +0.9 [-3.1, +5.6] | -1.9 [-4.3, +0.0] |
| Conspiracy theories & misinformation | 26 | 1.9 (0.0) | 11.9 (4.8) | 4.5 (1.6) | +9.9 [+4.2, +16.7] | -7.4 [-12.8, -2.6] |
| Weapons | 21 | 6.0 (2.4) | 19.4 (17.1) | 11.9 (8.7) | +13.5 [+6.0, +22.2] | -7.5 [-15.5, -0.8] |
| *Fewer than 20 prompts: listed, not interpreted* | | | | | | |
| Adult content | 19 | 10.5 (3.9) | 14.0 (9.2) | 7.5 (2.6) | +3.5 [-3.5, +11.4] | -6.6 [-11.8, -2.2] |
| Fraud & deception | 17 | 5.9 (1.5) | 12.7 (8.8) | 9.3 (5.4) | +6.9 [+2.0, +13.7] | -3.4 [-9.8, +2.9] |
| Substance abuse & banned substances | 16 | 6.2 (4.7) | 8.3 (5.2) | 5.2 (2.6) | +2.1 [-4.2, +8.3] | -3.1 [-8.9, +1.6] |
| Terrorism & organized crime | 16 | 7.8 (3.1) | 9.9 (5.7) | 4.2 (2.6) | +2.1 [-2.6, +6.2] | -5.7 [-10.4, -2.1] |
| Animal abuse | 11 | 2.3 (0.0) | 3.8 (2.3) | 0.8 (0.0) | +1.5 [-6.1, +9.8] | -3.0 [-7.6, +0.0] |
| Sexual exploitation & human trafficking | 11 | 15.9 (9.1) | 21.2 (12.1) | 10.6 (6.8) | +5.3 [-3.8, +13.6] | -10.6 [-18.9, -3.0] |
| Property crime & vandalism | 10 | 5.0 (0.0) | 13.3 (8.3) | 6.7 (5.0) | +8.3 [-2.5, +21.7] | -6.7 [-17.5, +2.5] |
| Soliciting personally identifiable information | 10 | 2.5 (2.5) | 6.7 (5.8) | 6.7 (5.0) | +4.2 [+0.0, +12.5] | +0.0 [+0.0, +0.0] |
| Self-harm | 8 | 3.1 (3.1) | 16.7 (8.3) | 5.2 (1.0) | +13.5 [+5.2, +21.9] | -11.5 [-19.8, -4.2] |
| Child abuse | 6 | 16.7 (4.2) | 20.8 (6.9) | 15.3 (4.2) | +4.2 [-2.8, +11.1] | -5.6 [-11.1, +0.0] |

#### English

| Category | Prompts | Base | Plain | SafeGrad | Plain against base | SafeGrad against plain |
|---|---:|---:|---:|---:|---|---|
| Violence & incitement | 80 | 3.8 (2.5) | 7.9 (6.2) | 2.9 (2.5) | +4.2 [+0.8, +8.3] | -5.0 [-9.6, -1.2] |
| Discrimination & injustice | 46 | 4.3 (2.2) | 4.3 (2.2) | 4.3 (1.4) | +0.0 [-6.5, +6.5] | +0.0 [+0.0, +0.0] |
| Hate speech & offensive language | 36 | 13.9 (5.6) | 8.3 (8.3) | 7.4 (5.6) | -5.6 [-13.9, +0.0] | -0.9 [-2.8, +0.0] |
| Bullying & harassment | 29 | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Non-violent unethical behavior (e.g. lying, cheating, etc.) | 29 | 6.9 (6.9) | 8.0 (3.4) | 0.0 (0.0) | +1.1 [-6.9, +10.3] | -8.0 [-18.4, +0.0] |
| Theft | 27 | 3.7 (0.0) | 3.7 (0.0) | 1.2 (0.0) | +0.0 [+0.0, +0.0] | -2.5 [-7.4, +0.0] |
| Conspiracy theories & misinformation | 26 | 0.0 (0.0) | 11.5 (1.3) | 0.0 (0.0) | +11.5 [+0.0, +26.9] | -11.5 [-26.9, +0.0] |
| Weapons | 21 | 4.8 (0.0) | 17.5 (15.9) | 11.1 (11.1) | +12.7 [+0.0, +28.6] | -6.3 [-17.5, +0.0] |
| *Fewer than 20 prompts: listed, not interpreted* | | | | | | |
| Adult content | 19 | 21.1 (10.5) | 5.3 (5.3) | 5.3 (1.8) | -15.8 [-36.8, +5.3] | +0.0 [-10.5, +10.5] |
| Fraud & deception | 17 | 0.0 (0.0) | 2.0 (2.0) | 3.9 (0.0) | +2.0 [+0.0, +5.9] | +2.0 [-5.9, +11.8] |
| Substance abuse & banned substances | 16 | 0.0 (0.0) | 0.0 (0.0) | 2.1 (2.1) | +0.0 [+0.0, +0.0] | +2.1 [+0.0, +6.2] |
| Terrorism & organized crime | 16 | 6.2 (0.0) | 10.4 (4.2) | 2.1 (0.0) | +4.2 [+0.0, +10.4] | -8.3 [-18.8, +0.0] |
| Animal abuse | 11 | 9.1 (0.0) | 0.0 (0.0) | 0.0 (0.0) | -9.1 [-27.3, +0.0] | +0.0 [+0.0, +0.0] |
| Sexual exploitation & human trafficking | 11 | 18.2 (18.2) | 9.1 (9.1) | 6.1 (0.0) | -9.1 [-27.3, +0.0] | -3.0 [-27.3, +18.2] |
| Property crime & vandalism | 10 | 0.0 (0.0) | 20.0 (20.0) | 13.3 (13.3) | +20.0 [+0.0, +50.0] | -6.7 [-20.0, +0.0] |
| Soliciting personally identifiable information | 10 | 10.0 (10.0) | 10.0 (10.0) | 10.0 (10.0) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Self-harm | 8 | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Child abuse | 6 | 33.3 (16.7) | 0.0 (0.0) | 11.1 (0.0) | -33.3 [-66.7, +0.0] | +11.1 [+0.0, +33.3] |

#### Chinese

| Category | Prompts | Base | Plain | SafeGrad | Plain against base | SafeGrad against plain |
|---|---:|---:|---:|---:|---|---|
| Violence & incitement | 80 | 6.2 (3.8) | 14.2 (10.4) | 7.5 (5.0) | +7.9 [+2.9, +13.8] | -6.7 [-12.9, -1.2] |
| Discrimination & injustice | 46 | 6.5 (0.0) | 9.4 (3.6) | 5.1 (1.4) | +2.9 [-4.3, +10.1] | -4.3 [-8.7, -1.4] |
| Hate speech & offensive language | 36 | 13.9 (2.8) | 14.8 (11.1) | 10.2 (6.5) | +0.9 [-8.3, +11.1] | -4.6 [-12.0, +0.9] |
| Bullying & harassment | 29 | 0.0 (0.0) | 11.5 (9.2) | 2.3 (1.1) | +11.5 [+1.1, +24.1] | -9.2 [-20.7, +0.0] |
| Non-violent unethical behavior (e.g. lying, cheating, etc.) | 29 | 10.3 (3.4) | 8.0 (5.7) | 10.3 (5.7) | -2.3 [-10.3, +3.4] | +2.3 [-3.4, +10.3] |
| Theft | 27 | 3.7 (3.7) | 3.7 (3.7) | 3.7 (3.7) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Conspiracy theories & misinformation | 26 | 3.8 (0.0) | 12.8 (5.1) | 9.0 (1.3) | +9.0 [-3.8, +23.1] | -3.8 [-12.8, +2.6] |
| Weapons | 21 | 4.8 (0.0) | 20.6 (15.9) | 14.3 (9.5) | +15.9 [+1.6, +33.3] | -6.3 [-20.6, +4.8] |
| *Fewer than 20 prompts: listed, not interpreted* | | | | | | |
| Adult content | 19 | 15.8 (5.3) | 21.1 (17.5) | 10.5 (3.5) | +5.3 [-10.5, +21.1] | -10.5 [-26.3, +0.0] |
| Fraud & deception | 17 | 11.8 (0.0) | 11.8 (5.9) | 11.8 (5.9) | +0.0 [-17.6, +17.6] | +0.0 [-17.6, +17.6] |
| Substance abuse & banned substances | 16 | 6.2 (6.2) | 8.3 (8.3) | 6.2 (6.2) | +2.1 [+0.0, +6.2] | -2.1 [-6.2, +0.0] |
| Terrorism & organized crime | 16 | 12.5 (6.2) | 14.6 (10.4) | 6.2 (6.2) | +2.1 [-10.4, +14.6] | -8.3 [-18.8, +0.0] |
| Animal abuse | 11 | 0.0 (0.0) | 9.1 (6.1) | 3.0 (0.0) | +9.1 [+0.0, +21.2] | -6.1 [-15.2, +0.0] |
| Sexual exploitation & human trafficking | 11 | 27.3 (9.1) | 18.2 (12.1) | 18.2 (12.1) | -9.1 [-27.3, +0.0] | +0.0 [+0.0, +0.0] |
| Property crime & vandalism | 10 | 10.0 (0.0) | 16.7 (10.0) | 10.0 (3.3) | +6.7 [-16.7, +33.3] | -6.7 [-33.3, +16.7] |
| Soliciting personally identifiable information | 10 | 0.0 (0.0) | 10.0 (6.7) | 6.7 (0.0) | +10.0 [+0.0, +30.0] | -3.3 [-10.0, +0.0] |
| Self-harm | 8 | 12.5 (12.5) | 16.7 (4.2) | 12.5 (0.0) | +4.2 [+0.0, +12.5] | -4.2 [-12.5, +0.0] |
| Child abuse | 6 | 16.7 (0.0) | 33.3 (22.2) | 16.7 (5.6) | +16.7 [+0.0, +50.0] | -16.7 [-50.0, +0.0] |

#### Swahili

| Category | Prompts | Base | Plain | SafeGrad | Plain against base | SafeGrad against plain |
|---|---:|---:|---:|---:|---|---|
| Violence & incitement | 80 | 1.2 (1.2) | 7.5 (2.9) | 5.0 (2.1) | +6.2 [+1.7, +11.7] | -2.5 [-7.9, +2.5] |
| Discrimination & injustice | 46 | 4.3 (0.0) | 1.4 (0.0) | 1.4 (0.0) | -2.9 [-10.1, +2.2] | +0.0 [-2.9, +3.6] |
| Hate speech & offensive language | 36 | 8.3 (0.0) | 10.2 (1.9) | 9.3 (3.7) | +1.9 [-9.3, +13.0] | -0.9 [-4.6, +2.8] |
| Bullying & harassment | 29 | 0.0 (0.0) | 3.4 (1.1) | 2.3 (0.0) | +3.4 [+0.0, +10.3] | -1.1 [-10.3, +6.9] |
| Non-violent unethical behavior (e.g. lying, cheating, etc.) | 29 | 3.4 (0.0) | 8.0 (1.1) | 5.7 (1.1) | +4.6 [+0.0, +12.6] | -2.3 [-10.3, +3.4] |
| Theft | 27 | 3.7 (0.0) | 3.7 (0.0) | 2.5 (0.0) | +0.0 [-11.1, +11.1] | -1.2 [-3.7, +0.0] |
| Conspiracy theories & misinformation | 26 | 3.8 (0.0) | 10.3 (0.0) | 5.1 (1.3) | +6.4 [+1.3, +12.8] | -5.1 [-12.8, +1.3] |
| Weapons | 21 | 4.8 (0.0) | 9.5 (7.9) | 7.9 (0.0) | +4.8 [-9.5, +19.0] | -1.6 [-14.3, +9.5] |
| *Fewer than 20 prompts: listed, not interpreted* | | | | | | |
| Adult content | 19 | 0.0 (0.0) | 5.3 (0.0) | 0.0 (0.0) | +5.3 [+0.0, +14.0] | -5.3 [-14.0, +0.0] |
| Fraud & deception | 17 | 5.9 (5.9) | 7.8 (2.0) | 5.9 (5.9) | +2.0 [-11.8, +17.6] | -2.0 [-17.6, +11.8] |
| Substance abuse & banned substances | 16 | 12.5 (6.2) | 8.3 (2.1) | 10.4 (0.0) | -4.2 [-18.8, +6.2] | +2.1 [-8.3, +18.8] |
| Terrorism & organized crime | 16 | 6.2 (0.0) | 6.2 (4.2) | 6.2 (4.2) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Animal abuse | 11 | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Sexual exploitation & human trafficking | 11 | 0.0 (0.0) | 21.2 (0.0) | 0.0 (0.0) | +21.2 [+6.1, +39.4] | -21.2 [-39.4, -6.1] |
| Property crime & vandalism | 10 | 10.0 (0.0) | 16.7 (3.3) | 3.3 (3.3) | +6.7 [-10.0, +30.0] | -13.3 [-33.3, +0.0] |
| Soliciting personally identifiable information | 10 | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Self-harm | 8 | 0.0 (0.0) | 4.2 (0.0) | 8.3 (4.2) | +4.2 [+0.0, +12.5] | +4.2 [-12.5, +25.0] |
| Child abuse | 6 | 0.0 (0.0) | 22.2 (0.0) | 0.0 (0.0) | +22.2 [+0.0, +44.4] | -22.2 [-44.4, +0.0] |

Swahili cells are the guard's raw verdicts. The guard missed our agreement bar in Swahili, and the translator-corrected numbers are not split by category.
