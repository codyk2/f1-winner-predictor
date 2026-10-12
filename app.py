"""Streamlit demo: pick a 2025 race, see each driver's pre-race win probability
next to what actually happened.

    make app        # or: streamlit run app.py

It only reads ``reports/test_2025/predictions.csv``, the output of the single
locked 2025 run. Nothing is refitted here.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from f1pred.config import REPORTS

BLUE, GREY = "#2a78d6", "#b9b8ae"
PREDICTIONS = REPORTS / "test_2025" / "predictions.csv"
MODEL, POLE = "XGBoost", "Pole sitter"


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS)


def ordinal(n: float) -> str:
    if pd.isna(n):
        return "did not start"
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def grid_label(g: float) -> str:
    if pd.isna(g):
        return "none"
    return "pit lane" if g > 20 else f"P{int(g)}"


st.set_page_config(page_title="F1 winner predictor · 2025", layout="wide")
st.title("F1 race winner predictor")
st.caption("Win probabilities computed before lights out for every 2025 Grand Prix. The model "
           "was trained on 2019–2024 and frozen before it saw any 2025 result.")

if not PREDICTIONS.exists():
    st.error("No 2025 predictions found. Run `make test-2025` first.")
    st.stop()

pred = load()
model = pred[pred["model"] == MODEL]
pole = pred[(pred["model"] == POLE) & (pred["rank"] == 1)].set_index("round")

races = model.drop_duplicates("round").set_index("round")["event"]
picks = model[model["rank"] == 1].set_index("round")
hits, pole_hits = int(picks["won"].sum()), int(pole["won"].sum())

top = st.columns(4)
top[0].metric("Races", len(races))
top[1].metric("Model picked the winner", f"{hits} of {len(races)}")
top[2].metric("Pole sitter won", f"{pole_hits} of {len(races)}")
top[3].metric("Winner in model's top 3",
              f"{int((model[model['won'] == 1]['rank'] <= 3).sum())} of {len(races)}")

rnd = st.selectbox(
    "Race", races.index,
    format_func=lambda r: f"Round {r} · {races[r]} · pick {'won' if picks.loc[r, 'won'] else 'lost'}")
race = model[model["round"] == rnd].sort_values("rank").copy()
pick, winner = race.iloc[0], race[race["won"] == 1].iloc[0]

verdict = "The pick won." if pick["won"] else (
    f"The pick finished {ordinal(pick['finish_pos'])}; {winner['FullName']} won from "
    f"{grid_label(winner['grid_pos'])} with a {winner['p']:.0%} pre-race chance "
    f"(ranked {int(winner['rank'])} by the model).")
st.subheader(f"{races[rnd]}: the model picked {pick['FullName']} at {pick['p']:.0%}")
st.write(f"{verdict} Pole sitter: {pole.loc[rnd, 'FullName']}.")

race["Result"] = race["won"].map({1: "Race winner", 0: "Other drivers"})
race["Finish"] = race["finish_pos"].map(ordinal)
race["Grid"] = race["grid_pos"].map(grid_label)
race["label"] = [f"{p:.0%}" if (r <= 3 or w) else ""
                 for p, r, w in zip(race["p"], race["rank"], race["won"], strict=True)]

base = alt.Chart(race).encode(
    y=alt.Y("Abbreviation:N", sort=race["Abbreviation"].tolist(), title=None,
            axis=alt.Axis(ticks=False, domain=False, labelPadding=8)),
    x=alt.X("p:Q", title="Pre-race win probability",
            axis=alt.Axis(format="%", grid=True, domain=False, ticks=False),
            scale=alt.Scale(domain=[0, max(0.6, float(race["p"].max()) + 0.1)])),
    tooltip=[alt.Tooltip("FullName:N", title="Driver"), alt.Tooltip("TeamId:N", title="Team"),
             alt.Tooltip("Grid:N"), alt.Tooltip("p:Q", title="Win probability", format=".1%"),
             alt.Tooltip("Finish:N", title="Finished")],
)
bars = base.mark_bar(size=13, cornerRadiusEnd=4).encode(
    color=alt.Color("Result:N", title=None,
                    scale=alt.Scale(domain=["Race winner", "Other drivers"], range=[BLUE, GREY]),
                    legend=alt.Legend(orient="top")))
labels = base.mark_text(align="left", dx=6, fontSize=12).encode(text="label:N")

left, right = st.columns([3, 2])
left.altair_chart((bars + labels).properties(height=24 * len(race) + 40),
                  width="stretch")
table = race[["rank", "FullName", "TeamId", "Grid", "p", "Finish"]].rename(columns={
    "rank": "Model rank", "FullName": "Driver", "TeamId": "Team", "p": "Win probability"})
right.dataframe(
    table, hide_index=True, width="stretch", height=24 * len(race) + 60,
    column_config={"Win probability": st.column_config.NumberColumn(format="percent")})

st.caption("Features use practice, qualifying, the sprint, the starting grid and earlier races "
           "only. Source: FastF1. See the README for validation and the full results table.")
