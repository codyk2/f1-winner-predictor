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
| 2 | pass 1 model | binary XGBoost, v1 parameters | 57.0 (56-58) | 0.0915 |  |  | start | 43.2 | 13.8 |
| 2 | monotone constraints | each pace and form feature may only push one way | 56.2 (55-57) | 0.0887 | +0.0028 | 3 of 4 | **yes** | 42.2 | 14.0 |
| 2 | softmax per race | trees trained on the race: one winner per race in the loss | 54.0 (52-55) | 0.0969 | -0.0082 | 1 of 4 | no | 41.5 | 12.5 |
| 2 | softmax + monotone | both of the above | 52.5 (52-53) | 0.0974 | -0.0088 | 1 of 4 | no | 40.7 | 11.8 |
| 2 | conditional logit, L2 1 | linear score per car, softmax over the field | 51.0 (51-51) | 0.1020 | -0.0133 | 0 of 4 | no | 40.0 | 11.0 |
| 2 | conditional logit, L2 10 | the same with a stronger penalty | 50.0 (50-50) | 0.0978 | -0.0091 | 1 of 4 | no | 39.0 | 11.0 |
| 2 | conditional logit from 2014 | trained on 2014 onwards, history in windows | 49.0 (49-49) | 0.1059 | -0.0172 | 0 of 4 | no | 39.0 | 10.0 |
| 2 | blend: trees x conditional logit | geometric mean of the two models | 53.0 (53-53) | 0.0928 | -0.0041 | 1 of 4 | no | 40.0 | 13.0 |
| 2 | depth 2, 200 trees | tree settings | 55.5 (55-56) | 0.0889 | -0.0003 | 2 of 4 | no | 41.5 | 14.0 |
| 2 | depth 2, 800 trees | tree settings | 56.3 (56-57) | 0.0904 | -0.0017 | 2 of 4 | no | 42.3 | 14.0 |
| 2 | depth 3, 200 trees | tree settings | 55.0 (55-55) | 0.0889 | -0.0002 | 2 of 4 | no | 41.0 | 14.0 |
| 2 | depth 3, 400 trees | tree settings | 56.7 (56-57) | 0.0892 | -0.0005 | 2 of 4 | no | 42.7 | 14.0 |
| 2 | temperature | one number fitted on earlier seasons flattens or sharpens | 56.2 (55-57) | 0.0905 | -0.0019 | 0 of 4 | no | 42.2 | 14.0 |
| 2b | pass 2 model | binary XGBoost with monotone constraints | 56.2 (55-57) | 0.0887 |  |  | start | 42.2 | 14.0 |
| 2b | softmax, 100 trees, child weight 5 | per-race softmax trees with monotone constraints | 53.8 (52-55) | 0.0945 | -0.0058 | 1 of 4 | no | 40.0 | 13.8 |
| 2b | softmax, 200 trees, child weight 5 | per-race softmax trees with monotone constraints | 52.7 (52-54) | 0.0938 | -0.0051 | 1 of 4 | no | 39.5 | 13.2 |
| 2b | softmax, 200 trees, child weight 1 | per-race softmax trees with monotone constraints | 50.7 (50-51) | 0.0955 | -0.0068 | 1 of 4 | no | 37.7 | 13.0 |
| 2b | softmax, 800 trees, child weight 5 | per-race softmax trees with monotone constraints | 52.2 (52-53) | 0.1028 | -0.0141 | 0 of 4 | no | 41.0 | 11.2 |
| 2b | conditional logit, L2 30 | stronger penalty | 51.0 (51-51) | 0.0989 | -0.0102 | 0 of 4 | no | 42.0 | 9.0 |
| 2b | conditional logit, L2 100 | stronger penalty | 49.0 (49-49) | 0.1045 | -0.0158 | 0 of 4 | no | 41.0 | 8.0 |
| 2b | conditional logit, L2 300 | stronger penalty | 47.0 (47-47) | 0.1157 | -0.0270 | 0 of 4 | no | 40.0 | 7.0 |
| 2b | blend: monotone trees x logit L2 100 | geometric mean of the two models | 55.3 (54-57) | 0.0916 | -0.0029 | 2 of 4 | no | 43.8 | 11.5 |
| 2b | monotone trees from 2014 | 2014-2018 as training rows, history in windows | 55.8 (55-56) | 0.0889 | -0.0002 | 1 of 4 | no | 41.8 | 14.0 |
