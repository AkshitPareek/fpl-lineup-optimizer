"""
Offline accuracy check for PointPredictor against completed seasons.

Rebuilds the player state as it stood before each gameweek deadline from the
public per-gameweek archive (github.com/vaastav/Fantasy-Premier-League), runs
the predictor, and compares its projections with the points actually scored.

Usage:
    python predictor_eval.py --season 2025-26 --horizons 1 5 8
"""

import argparse
import importlib.util
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache", "seasons")


def previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[2:]}"


def _read(season: str, name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, season, os.path.basename(name))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.read_csv(f"{ARCHIVE_URL}/{season}/{name}").to_csv(path, index=False)
    return pd.read_csv(path)


class SeasonArchive:
    """One completed season, able to reproduce the pre-deadline state of any gameweek."""

    def __init__(self, season: str):
        self.season = season
        self.gws = _read(season, "gws/merged_gw.csv")
        self.players = _read(season, "players_raw.csv")
        self.teams = _read(season, "teams.csv")
        fixtures = _read(season, "fixtures.csv")
        self.fixtures = [
            {
                "event": int(f.event),
                "team_h": int(f.team_h),
                "team_a": int(f.team_a),
                "team_h_difficulty": int(f.team_h_difficulty),
                "team_a_difficulty": int(f.team_a_difficulty),
            }
            for f in fixtures.itertuples()
            if not pd.isna(f.event)
        ]
        team_ids = dict(zip(self.teams["name"], self.teams["id"]))
        self.gws["team_id"] = self.gws["team"].map(team_ids)
        for col in ["expected_goals", "expected_assists", "expected_goals_conceded", "influence", "ict_index"]:
            self.gws[col] = pd.to_numeric(self.gws[col], errors="coerce").fillna(0.0)

        per_round = self.gws.groupby(["element", "round"])["total_points"].sum()
        self.actual = per_round.unstack(fill_value=0).reindex(columns=range(1, 39), fill_value=0)

    def season_totals_by_code(self) -> pd.DataFrame:
        """End-of-season totals keyed by the cross-season player code."""
        cols = {"total_points": "prior_points", "minutes": "prior_minutes", "starts": "prior_starts"}
        return self.players.set_index("code")[list(cols)].rename(columns=cols)

    def state_before(self, gameweek: int, prior: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        past = self.gws[self.gws["round"] < gameweek]
        rows = []
        for pid, g in past.sort_values(["round", "kickoff_time"]).groupby("element"):
            played = g[g["minutes"] > 0]
            last_4_rounds = sorted(g["round"].unique())[-4:]
            last_4_played = played.groupby("round")["total_points"].sum().tail(4)
            rows.append({
                "id": pid,
                "team": int(g["team_id"].iloc[-1]),
                "now_cost": int(g["value"].iloc[-1]),
                "minutes": int(g["minutes"].sum()),
                "starts": int(g["starts"].sum()),
                "total_points": int(g["total_points"].sum()),
                "goals_scored": int(g["goals_scored"].sum()),
                "assists": int(g["assists"].sum()),
                "expected_goals": float(g["expected_goals"].sum()),
                "expected_assists": float(g["expected_assists"].sum()),
                "expected_goals_conceded": float(g["expected_goals_conceded"].sum()),
                "influence": float(g["influence"].sum()),
                "ict_index": float(g["ict_index"].sum()),
                "points_per_game": g["total_points"].sum() / max(1, played["round"].nunique()),
                "form": last_4_played.mean() if len(last_4_played) else 0.0,
                "recent_4_minutes": int(g[g["round"].isin(last_4_rounds)]["minutes"].sum()),
            })
        stats = pd.DataFrame(rows).set_index("id")
        state = self.players.set_index("id")[
            ["code", "web_name", "element_type", "team", "now_cost", "selected_by_percent"]
        ].copy()
        state["status"] = "a"
        state["chance_of_playing_next_round"] = None
        for col in stats.columns:
            if col in state.columns:
                state[col] = stats[col].combine_first(state[col]).astype(int)
            else:
                state[col] = stats[col].reindex(state.index).fillna(0)
        state = state.reset_index()
        if prior is not None:
            state = state.merge(prior, left_on="code", right_index=True, how="left")
        return state


def load_predictor_class(module_path: Optional[str]):
    if not module_path:
        from point_predictor import PointPredictor
        return PointPredictor
    spec = importlib.util.spec_from_file_location("predictor_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PointPredictor


def evaluate(season: str, horizons: List[int], start_gw: int = 5, end_gw: int = 30,
             predictor_cls=None, use_prior: bool = True) -> Dict[int, Dict[str, float]]:
    """
    Score projections over each horizon K for players who started regularly
    in the four gameweeks before the deadline.

    Returns per horizon: correlation with actual points, mean absolute error
    per gameweek, and the actual points per gameweek of the 40 highest
    projected players.
    """
    predictor_cls = predictor_cls or load_predictor_class(None)
    archive = SeasonArchive(season)
    prior = SeasonArchive(previous_season(season)).season_totals_by_code() if use_prior else None
    teams = archive.teams
    max_k = max(horizons)

    scores = {k: {"corr": [], "mae": [], "top40": []} for k in horizons}
    for gw in range(start_gw, min(end_gw, 39 - max_k) + 1):
        state = archive.state_before(gw, prior)
        predictor = predictor_cls(state, teams, archive.fixtures, gw, use_understat=False)
        preds = predictor.predict_all_players(gameweeks=max_k).set_index("id")
        regular = state.set_index("id")["recent_4_minutes"] >= 240
        ids = [i for i in regular[regular].index if i in archive.actual.index and i in preds.index]
        for k in horizons:
            cols = [f"xp_gw{g}" for g in range(gw, gw + k)]
            projected = preds.loc[ids, cols].sum(axis=1)
            actual = archive.actual.loc[ids, list(range(gw, gw + k))].sum(axis=1)
            scores[k]["corr"].append(np.corrcoef(projected, actual)[0, 1])
            scores[k]["mae"].append((projected - actual).abs().mean() / k)
            scores[k]["top40"].append(actual[projected.nlargest(40).index].mean() / k)
    return {k: {m: float(np.mean(v)) for m, v in s.items()} for k, s in scores.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 8])
    parser.add_argument("--start-gw", type=int, default=5)
    parser.add_argument("--end-gw", type=int, default=30)
    parser.add_argument("--predictor", help="Path to an alternative point_predictor.py to score")
    parser.add_argument("--no-prior", action="store_true", help="Withhold last season's totals")
    args = parser.parse_args()

    results = evaluate(args.season, args.horizons, args.start_gw, args.end_gw,
                       load_predictor_class(args.predictor), use_prior=not args.no_prior)
    print(f"Season {args.season}, deadlines GW{args.start_gw}-{args.end_gw}")
    print(f"{'horizon':>8} {'corr':>7} {'mae/gw':>7} {'top40 pts/gw':>13}")
    for k, r in results.items():
        print(f"{k:>8} {r['corr']:>7.3f} {r['mae']:>7.2f} {r['top40']:>13.2f}")


if __name__ == "__main__":
    main()
