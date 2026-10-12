## v2_dev

| Model | Picked the winner | 95% CI | vs pole sitter (95% CI) | Winner in top 3 | Log loss | Race log loss |
| --- | --- | --- | --- | --- | --- | --- |
| v2.0 | 56 of 92 (60.9%) | 50.7% to 70.2% | +5.4% (-4.3% to +15.2%) | 85 of 92 | 0.0887 | 1.079 |
| v1.0 | 54 of 92 (58.7%) | 48.5% to 68.2% | +3.3% (-5.4% to +13.0%) | 84 of 92 | 0.0946 | 1.158 |
| Pole sitter | 51 of 92 (55.4%) | 45.3% to 65.2% | baseline | 80 of 92 | n/a | n/a |
| Grid-only logistic regression | 50 of 92 (54.3%) | 44.2% to 64.1% | -1.1% (-3.3% to +0.0%) | 81 of 92 | 0.1178 | 1.551 |

## v2_dev_folds

| Fold | Races | v2.0 | v1.0 | Pole sitter | Grid-only logistic regression |
| --- | --- | --- | --- | --- | --- |
| 2022 | 22 | 10 | 9 | 10 | 9 |
| 2023 | 22 | 19 | 18 | 14 | 14 |
| 2024 | 24 | 13 | 12 | 11 | 11 |
| 2025 | 24 | 14 | 15 | 16 | 16 |
| all four | 92 | 56 | 54 | 51 | 50 |

## v2_dev_notes

v2.0 as frozen: 15 features, binary XGBoost (`max_depth` 2, 400 trees) with monotone constraints, 15 seeds averaged. Against v1.0 on these 92 races: winners +2.2% (95% CI -5.4% to +9.8%), log loss -0.0059 (95% CI -0.0122 to +0.0010). These are development numbers.
