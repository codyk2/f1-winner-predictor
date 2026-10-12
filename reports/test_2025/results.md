## results_2025

| Model | Picked the winner | 95% CI | vs pole sitter (95% CI) | Winner in top 3 | Hits in the 8 races pole lost | Log loss | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 15 of 24 (62.5%) | 41.7% to 83.3% | -4.2% (-20.8% to +12.5%) | 24 of 24 (100.0%) | 2 of 8 | 0.0895 | 0.0285 |
| Pole sitter | 16 of 24 (66.7%) | 50.0% to 83.3% | baseline | 23 of 24 (95.8%) | 0 of 8 | n/a | n/a |
| Form leader | 7 of 24 (29.2%) | 12.5% to 45.8% | -37.5% (-62.5% to -8.3%) | 17 of 24 (70.8%) | 3 of 8 | n/a | n/a |
| Grid-only logistic regression | 16 of 24 (66.7%) | 50.0% to 83.3% | same result in every race | 23 of 24 (95.8%) | 0 of 8 | 0.0985 | 0.0299 |

## results_2025_notes

Log loss against the grid-only logistic regression: -0.0090 (-9.2%; 95% CI -0.0221 to +0.0053). Expected calibration error on 2025: 0.022. Intervals are 95% percentile bootstraps over the 24 races (10,000 resamples); one race is about 4%.

## per_race_2025

| Rd | Race | Winner (grid) | Model pick (grid) | P(pick) | Pole sitter | Pick right? | P(winner) | Winner's rank | Pick driven by | Miss cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Australian | NOR (P1) | NOR (P1) | 54% | NOR | yes | 54% | 1 | starting position, qualifying position |  |
| 2 | Chinese | PIA (P1) | PIA (P1) | 36% | PIA | yes | 36% | 1 | starting position, qualifying position |  |
| 3 | Japanese | VER (P1) | VER (P1) | 38% | VER | yes | 38% | 1 | starting position, qualifying position |  |
| 4 | Bahrain | PIA (P1) | PIA (P1) | 42% | PIA | yes | 42% | 1 | starting position, qualifying position |  |
| 5 | Saudi Arabian | PIA (P2) | PIA (P2) | 42% | VER | yes | 42% | 1 | starting position, qualifying gap to pole |  |
| 6 | Miami | PIA (P4) | NOR (P2) | 34% | VER | **no** | 32% | 2 | starting position, recent finishes | Race incident |
| 7 | Emilia Romagna | VER (P2) | PIA (P1) | 71% | PIA | **no** | 9% | 3 | recent wins, starting position | Pace misread |
| 8 | Monaco | NOR (P1) | PIA (P3) | 39% | NOR | **no** | 39% | 2 | recent wins, recent finishes | Pace misread |
| 9 | Spanish | PIA (P1) | PIA (P1) | 72% | PIA | yes | 72% | 1 | recent wins, starting position |  |
| 10 | Canadian | RUS (P1) | PIA (P3) | 59% | RUS | **no** | 24% | 2 | recent wins, qualifying gap to pole | Pace misread |
| 11 | Austrian | NOR (P1) | NOR (P1) | 47% | NOR | yes | 47% | 1 | starting position, qualifying position |  |
| 12 | British | NOR (P3) | PIA (P2) | 39% | VER | **no** | 27% | 2 | starting position, recent finishes | Race incident |
| 13 | Belgian | PIA (P2) | NOR (P1) | 62% | NOR | **no** | 13% | 3 | recent wins, starting position | Pace misread |
| 14 | Hungarian | NOR (P3) | LEC (P1) | 36% | LEC | **no** | 27% | 2 | starting position, qualifying position | Safety car or strategy |
| 15 | Dutch | PIA (P1) | PIA (P1) | 47% | PIA | yes | 47% | 1 | starting position, qualifying position |  |
| 16 | Italian | VER (P1) | NOR (P2) | 44% | VER | **no** | 24% | 2 | recent wins, starting position | Pace misread |
| 17 | Azerbaijan | VER (P1) | VER (P1) | 56% | VER | yes | 56% | 1 | starting position, qualifying position |  |
| 18 | Singapore | RUS (P1) | RUS (P1) | 52% | RUS | yes | 52% | 1 | starting position, qualifying position |  |
| 19 | United States | VER (P1) | VER (P1) | 47% | VER | yes | 47% | 1 | starting position, qualifying position |  |
| 20 | Mexico City | NOR (P1) | NOR (P1) | 49% | NOR | yes | 49% | 1 | starting position, qualifying position |  |
| 21 | São Paulo | NOR (P1) | NOR (P1) | 30% | NOR | yes | 30% | 1 | starting position, qualifying position |  |
| 22 | Las Vegas | VER (P2) | NOR (P1) | 61% | NOR | **no** | 28% | 2 | qualifying position, starting position | Race incident |
| 23 | Qatar | VER (P3) | VER (P3) | 35% | PIA | yes | 35% | 1 | recent finishes, qualifying gap to pole |  |
| 24 | Abu Dhabi | VER (P1) | VER (P1) | 60% | VER | yes | 60% | 1 | recent wins, starting position |  |

## miss_causes

| Miss cause | Races |
| --- | --- |
| Race incident | 3 |
| Safety car or strategy | 1 |
| Reliability | 0 |
| Pace misread | 5 |

## miss_table

| Rd | Race | Pick | Winner | Cause | What happened |
| --- | --- | --- | --- | --- | --- |
| 6 | Miami | NOR (P2) | PIA (P4) | Race incident | Fell from 2nd to 6th on lap 1 after running wide fighting Verstappen for the lead; recovered to 2nd behind team-mate Piastri. |
| 7 | Emilia Romagna | PIA (P1) | VER (P2) | Pace misread | Verstappen took the lead at the first chicane and led all 63 laps. The model's 71% for Piastri leaned on his four wins in five races. |
| 8 | Monaco | PIA (P3) | NOR (P1) | Pace misread | Picked Piastri from 3rd on form over pole sitter Norris (39% each). The top three finished in grid order. |
| 10 | Canadian | PIA (P3) | RUS (P1) | Pace misread | Picked Piastri from 3rd on form; pole sitter Russell led 43 of 70 laps and won. Piastri finished 4th. |
| 12 | British | PIA (P2) | NOR (P3) | Race incident | Led 36 of 52 laps, then a 10-second penalty for a safety-car infringement handed the win to Norris. |
| 13 | Belgian | NOR (P1) | PIA (P2) | Pace misread | Wet safety-car start; Piastri passed Norris on the first green lap and led to the flag. Equal cars: 62% against 13% was too confident. |
| 14 | Hungarian | LEC (P1) | NOR (P3) | Safety car or strategy | Leclerc led 28 laps on a two-stop and faded to 4th; Norris, 5th after lap 1, won with a single stop on lap 31. |
| 16 | Italian | NOR (P2) | VER (P1) | Pace misread | Picked Norris from 2nd on form over pole sitter Verstappen, who led 42 of 53 laps and won. |
| 22 | Las Vegas | NOR (P1) | VER (P2) | Race incident | Ran wide at turn 1 and fell from pole to 3rd on lap 1; Verstappen led all 50 laps. Both McLarens were later disqualified. |

## resume_bullets

- Built a FastF1 data pipeline ingesting practice, qualifying, sprint and race data for 152 races (2019-2025), engineering 19 pre-race features including long-run practice pace, qualifying gap and rolling driver/team form
- Trained an XGBoost win-probability model with season-by-season walk-forward validation and automated leakage tests ensuring only pre-race information
- Reached a 100% top-3 hit rate on 24 held-out 2025 races and called 2 of the 8 won from off pole; 9% lower log loss than a grid-only model
