| Model | Picked the winner | 95% CI | vs pole sitter (95% CI) | Winner in top 3 | Hits when pole lost | Log loss | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 13 of 17 (76.5%) | 52.7% to 90.4% | +5.9% (+0.0% to +17.6%) | 16 of 17 (73.0% to 99.0%) | 1 of 5 | 0.0686 | 0.0203 |
| Pole sitter | 12 of 17 (70.6%) | 46.9% to 86.7% | baseline | 16 of 17 (73.0% to 99.0%) | 0 of 5 | n/a | n/a |
| Form leader | 1 of 17 (5.9%) | 1.0% to 27.0% | -64.7% (-88.2% to -41.2%) | 9 of 17 (31.0% to 73.8%) | 0 of 5 | n/a | n/a |
| Grid-only logistic regression | 12 of 17 (70.6%) | 46.9% to 86.7% | same hits and misses | 16 of 17 (73.0% to 99.0%) | 0 of 5 | 0.1028 | 0.0254 |

Rounds 1-17 of 2026, re-scored on 2026-10-12 after the races were run. Each race was scored by the frozen v1.0 setup retrained on every earlier race, with that race's result hidden. Log loss against the grid-only logistic regression: -0.0342 (-33.3%; 95% CI -0.0835 to +0.0006). The single most favourable race (round 13 of 2026) supplies 65% of that gap; without it the gap is -16.4%.

| Rd | Race | Winner (grid) | Model pick (grid) | P(pick) | Pole sitter | Pick right? | Pole right? | P(winner) | Winner's rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Australian | RUS (P1) | RUS (P1) | 70% | RUS | yes | yes | 70% | 1 |
| 2 | Chinese | ANT (P1) | ANT (P1) | 64% | ANT | yes | yes | 64% | 1 |
| 3 | Japanese | ANT (P1) | ANT (P1) | 64% | ANT | yes | yes | 64% | 1 |
| 4 | Miami | ANT (P1) | ANT (P1) | 70% | ANT | yes | yes | 70% | 1 |
| 5 | Canadian | ANT (P2) | RUS (P1) | 50% | RUS | **no** | no | 32% | 2 |
| 6 | Monaco | ANT (P1) | ANT (P1) | 68% | ANT | yes | yes | 68% | 1 |
| 7 | Barcelona | HAM (P2) | ANT (P3) | 50% | RUS | **no** | no | 20% | 3 |
| 8 | Austrian | RUS (P1) | RUS (P1) | 33% | RUS | yes | yes | 33% | 1 |
| 9 | British | LEC (P2) | ANT (P1) | 78% | ANT | **no** | no | 6% | 3 |
| 10 | Belgian | ANT (P1) | ANT (P1) | 74% | ANT | yes | yes | 74% | 1 |
| 11 | Hungarian | NOR (P1) | NOR (P1) | 42% | NOR | yes | yes | 42% | 1 |
| 12 | Dutch | NOR (P1) | NOR (P1) | 62% | NOR | yes | yes | 62% | 1 |
| 13 | Italian | ANT (P19) | HAM (P4) | 23% | GAS | **no** | no | 16% | 4 |
| 14 | Spanish | ANT (P2) | ANT (P2) | 40% | NOR | yes | no | 40% | 1 |
| 15 | Azerbaijan | RUS (P1) | RUS (P1) | 59% | RUS | yes | yes | 59% | 1 |
| 16 | Bahrain | VER (P1) | VER (P1) | 48% | VER | yes | yes | 48% | 1 |
| 17 | Singapore | VER (P1) | VER (P1) | 47% | VER | yes | yes | 47% | 1 |
