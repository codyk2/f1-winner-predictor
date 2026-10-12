"""A small fake championship with the same raw-table schema as ``f1pred.ingest``.

It exists so the leakage tests run anywhere (including CI, where the real data
is not checked in) and it deliberately contains the awkward cases: a sprint
weekend, a weekend with no practice, pit-lane starts, DNFs, a driver with no
qualifying time, a team that is renamed between seasons, a venue whose name
changes, practice laps with no DriverId and a practice-only driver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASONS = (2021, 2022, 2023)
ROUNDS = 6
POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]
VENUES = ["Sakhir", "Monte Carlo", "Silverstone", "Monza", "Baku", "Suzuka"]
TEAMS = ["mercedes", "ferrari", "red_bull", "mclaren", "toro_rosso"]
RENAMES = {2022: {"toro_rosso": "alphatauri"}, 2023: {"toro_rosso": "rb"}}
CONVENTIONAL = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
SPRINT = ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]


def _td(seconds: float) -> pd.Timedelta:
    return pd.to_timedelta(round(float(seconds), 3), unit="s")


def make_raw(seed: int = 0, seasons: tuple[int, ...] = SEASONS) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    drivers = [{"DriverId": f"driver_{i:02d}", "DriverNumber": str(i + 1),
                "Abbreviation": f"D{i:02d}", "FullName": f"Driver {i:02d}",
                "team": TEAMS[i // 2], "skill": i * 0.12} for i in range(10)]
    events, race, quali, sprint, laps, weather = [], [], [], [], [], []

    for season in seasons:
        for rnd in range(1, ROUNDS + 1):
            is_sprint = rnd == 3
            names = SPRINT if is_sprint else CONVENTIONAL
            location = VENUES[rnd - 1]
            if location == "Monte Carlo" and season >= 2022:
                location = "Monaco"  # same circuit, new spelling
            events.append({
                "season": season, "round": rnd, "event": f"{VENUES[rnd - 1]} Grand Prix",
                "location": location, "country": "Nowhere",
                "date": pd.Timestamp(season, 3, 1) + pd.to_timedelta(14 * rnd, unit="D"),
                "format": "sprint_qualifying" if is_sprint else "conventional",
                **{f"session{i}": n for i, n in enumerate(names, start=1)},
            })
            entry = [dict(d) for d in drivers]
            for d in entry:
                d["TeamId"] = RENAMES.get(season, {}).get(d["team"], d["team"])
            if season == 2023 and rnd >= 4:  # mid-season driver change
                entry[9].update(DriverId="driver_99", DriverNumber="99",
                                Abbreviation="D99", FullName="Driver 99", skill=1.0)

            def ident(d: dict) -> dict:
                return {k: d[k] for k in ("DriverNumber", "Abbreviation", "DriverId",
                                          "FullName", "TeamId")} | {"TeamName": d["TeamId"]}

            # qualifying
            pace = np.array([90 + d["skill"] + rng.normal(0, 0.25) for d in entry])
            no_time = 7 if rnd == 2 else None  # this driver sets no qualifying time
            order = np.argsort(np.where(np.arange(10) == no_time, 999, pace))
            for pos, i in enumerate(order, start=1):
                timed = i != no_time
                quali.append({
                    **ident(entry[i]), "GridPosition": np.nan, "Position": float(pos),
                    "ClassifiedPosition": pd.NA, "Status": pd.NA, "Points": np.nan,
                    "Laps": np.nan,
                    "Q1": _td(pace[i] + 0.6) if timed else pd.NaT,
                    "Q2": _td(pace[i] + 0.3) if timed and pos <= 7 else pd.NaT,
                    "Q3": _td(pace[i]) if timed and pos <= 5 else pd.NaT,
                    "season": season, "round": rnd, "session": "Qualifying",
                })

            # sprint
            if is_sprint:
                s_order = np.argsort(pace + rng.normal(0, 0.3, 10))
                for pos, i in enumerate(s_order, start=1):
                    sprint.append({
                        **ident(entry[i]), "GridPosition": float(pos), "Position": float(pos),
                        "ClassifiedPosition": str(pos), "Status": "Finished",
                        "Points": float(SPRINT_POINTS[pos - 1]) if pos <= 8 else 0.0,
                        "Laps": 17.0, "Q1": pd.NaT, "Q2": pd.NaT, "Q3": pd.NaT,
                        "season": season, "round": rnd, "session": "Sprint",
                    })

            # race: grid from qualifying, one pit-lane start on even rounds
            grid = {int(i): float(pos) for pos, i in enumerate(order, start=1)}
            if rnd % 2 == 0:
                grid[int(order[4])] = 0.0
            race_pace = pace + rng.normal(0, 0.5, 10)
            retired = set(rng.choice(10, size=int(rng.integers(0, 3)), replace=False).tolist())
            finish = sorted(range(10), key=lambda i: (i in retired, race_pace[i]))
            for pos, i in enumerate(finish, start=1):
                out = i in retired
                race.append({
                    **ident(entry[i]), "GridPosition": grid[i], "Position": float(pos),
                    "ClassifiedPosition": "R" if out else str(pos),
                    "Status": "Retired" if out else "Finished",
                    "Points": 0.0 if out or pos > 10 else float(POINTS[pos - 1]),
                    "Laps": 20.0 if out else 57.0, "Q1": pd.NaT, "Q2": pd.NaT, "Q3": pd.NaT,
                    "season": season, "round": rnd, "session": "Race",
                })

            # weather for qualifying and practice
            sessions = [n for n in names if n.startswith("Practice")] + ["Qualifying"]
            wet_quali = rnd == 5
            for name in sessions:
                for k in range(4):
                    weather.append({
                        "Time": pd.to_timedelta(15 * k, unit="min"),
                        "AirTemp": 22 + rng.normal(0, 1),
                        "TrackTemp": 30 + rnd + rng.normal(0, 1),
                        "Rainfall": bool(wet_quali and name == "Qualifying" and k == 2),
                        "Humidity": 50.0, "WindSpeed": 2.0,
                        "season": season, "round": rnd, "session": name,
                    })

            # practice laps: nothing at all for round 6 (session lost to weather)
            if rnd == 6:
                continue
            for name in (n for n in names if n.startswith("Practice")):
                reserve = {"DriverId": pd.NA, "DriverNumber": "40", "Abbreviation": "RES",
                           "skill": 1.5, "TeamId": "mercedes"}
                runners = entry + [reserve] * (name == "Practice 1")
                for j, d in enumerate(runners):
                    # one session loses its DriverId lookup, as when Ergast is unavailable
                    driver_id = pd.NA if (season == 2022 and rnd == 2) else d["DriverId"]
                    base = 92 + d["skill"] + rng.normal(0, 0.15)
                    stints = [(1, "SOFT", 4, -1.5)]
                    if (j + rnd) % 4 != 0:  # a quarter of the field does no long run
                        stints.append((2, "MEDIUM" if j % 2 else "HARD", 10, 1.0))
                    lap_no = 0
                    for stint, compound, n_laps, offset in stints:
                        for k in range(n_laps):
                            lap_no += 1
                            first, last = k == 0, k == n_laps - 1
                            slow = 12.0 if first or last else (5.0 if k == 4 else 0.0)
                            laps.append({
                                "Driver": d["Abbreviation"], "DriverNumber": d["DriverNumber"],
                                "Team": d["TeamId"], "LapNumber": float(lap_no),
                                "LapTime": _td(base + offset + slow + rng.normal(0, 0.1)),
                                "Stint": float(stint), "Compound": compound,
                                "TyreLife": float(k + 1), "FreshTyre": True,
                                "TrackStatus": "12" if (k == 2 and j == 3) else "1",
                                "IsAccurate": not (first or last),
                                "PitInTime": _td(600 + 100 * lap_no) if last else pd.NaT,
                                "PitOutTime": _td(500 + 100 * lap_no) if first else pd.NaT,
                                "DriverId": driver_id,
                                "season": season, "round": rnd, "session": name,
                            })

    string_cols = ["DriverNumber", "Abbreviation", "DriverId", "FullName", "TeamName", "TeamId",
                   "ClassifiedPosition", "Status", "Driver", "Team", "Compound", "TrackStatus"]

    def frame(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        for col in string_cols:
            if col in df:
                df[col] = df[col].astype("string")
        for col in ("Q1", "Q2", "Q3", "LapTime", "PitInTime", "PitOutTime"):
            if col in df:
                df[col] = pd.to_timedelta(df[col].astype(object))
        return df

    return {"events": frame(events), "race": frame(race), "quali": frame(quali),
            "sprint": frame(sprint), "laps": frame(laps), "weather": frame(weather)}
