| Pass | Candidate | What changed | Winners (of 92) | Log loss | Gain | Folds better | Kept | 2022-24 (of 68) | 2025 (of 24) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Pole sitter | reference | 51 | n/a |  |  |  | 35 | 16 |
|  | Grid-only logistic regression | reference | 50 | 0.1178 |  |  |  | 34 | 16 |
| 1 | v1.0 as frozen | 14 features, trained on 2019 onwards | 54.0 (53-55) | 0.0950 |  |  | start | 39.0 | 15.0 |
| 1 | history in windows | form and circuit windows see 2014-2018 results | 53.3 (52-55) | 0.0945 | +0.0005 | 2 of 4 | no | 38.2 | 15.2 |
| 1 | train from 2014 | 2014-2018 races added as training rows | 54.3 (53-55) | 0.0959 | -0.0008 | 1 of 4 | no | 39.8 | 14.5 |
| 1 | + quali | added: quali_seg_gap_pct, quali_margin_pct | 52.7 (52-53) | 0.0937 | +0.0013 | 4 of 4 | **yes** | 39.7 | 13.0 |
| 1 | + pace_form | added: drv_quali_gap_avg_5, quali_gap_vs_form, team_quali_gap, team_quali_gap_avg_5, team_gap_vs_form | 50.7 (50-51) | 0.0964 | -0.0027 | 1 of 4 | no | 38.2 | 12.5 |
| 1 | + practice | replaces longrun_gap_pct, has_longrun: longrun_gap_v2 | 57.0 (56-58) | 0.0915 | +0.0022 | 4 of 4 | **yes** | 43.2 | 13.8 |
| 1 | + sticky | added: track_pole_win_eb, track_grid_corr_eb | 54.8 (54-56) | 0.0912 | +0.0003 | 3 of 4 | no | 41.8 | 13.0 |
| 1 | + season | added: season_round, new_regs | 55.8 (55-56) | 0.0916 | -0.0001 | 2 of 4 | no | 42.8 | 13.0 |
