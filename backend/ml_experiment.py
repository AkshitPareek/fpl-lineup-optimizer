"""
Experiment: a gradient-boosted correction on top of PointPredictor.

For every deadline of past seasons it records the engine's projection for each of
the next eight weeks alongside point-in-time player stats and fixture details,
then trains a model on all other seasons and scores the held-out one. The
blend (half engine, half model) projects more accurately than the engine in all
three seasons tested, but replayed seasons did not score more points, so it is
not used by the API.

Needs scikit-learn, which the API does not install:
    pip install scikit-learn
    python ml_experiment.py build 2023-24 2024-25 2025-26
    python ml_experiment.py evaluate 2023-24 2024-25 2025-26
"""
import argparse
import os

import numpy as np
import pandas as pd

import point_predictor
point_predictor.UNDERSTAT_AVAILABLE = False  # current Understat totals would leak later weeks
import predictor_eval as pe
from point_predictor import PointPredictor

ROWS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache", "ml")
HORIZON = 8
PLAYING_MINUTES = 180  # trained on players with this many minutes in their last four games


def build_rows(season: str) -> pd.DataFrame:
    archive = pe.SeasonArchive(season)
    prior = pe.SeasonArchive(pe.previous_season(season)).season_totals_by_code()
    teams = archive.teams.copy()
    teams[["strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"]] = 0

    fixtures_by = {}
    for f in archive.fixtures:
        fixtures_by.setdefault((f["team_h"], f["event"]), []).append((1, f["team_h_difficulty"]))
        fixtures_by.setdefault((f["team_a"], f["event"]), []).append((0, f["team_a_difficulty"]))

    rows = []
    for gw in range(3, 39):
        state = archive.state_before(gw, prior)
        k_max = min(HORIZON, 39 - gw)
        preds = PointPredictor(state, teams, archive.fixtures, gw, use_understat=False).predict_all_players(gameweeks=k_max).set_index("id")
        st = state.set_index("id")
        for pid, p in st.iterrows():
            if pid not in preds.index:
                continue
            for k in range(k_max):
                w = gw + k
                fx = fixtures_by.get((p["team"], w), [])
                rows.append({
                    "season": season, "gw": gw, "k": k, "id": pid,
                    "engine_xp": preds.at[pid, f"xp_gw{w}"],
                    "element_type": p["element_type"], "now_cost": p["now_cost"],
                    "minutes": p["minutes"], "starts": p["starts"], "total_points": p["total_points"],
                    "goals_scored": p["goals_scored"], "assists": p["assists"],
                    "expected_goals": p["expected_goals"], "expected_assists": p["expected_assists"],
                    "expected_goals_conceded": p["expected_goals_conceded"],
                    "influence": p["influence"], "ict_index": p["ict_index"],
                    "recent_4_minutes": p["recent_4_minutes"],
                    "prior_points": p.get("prior_points"), "prior_minutes": p.get("prior_minutes"), "prior_starts": p.get("prior_starts"),
                    "fixtures": len(fx), "home": sum(h for h, _ in fx), "difficulty": sum(d for _, d in fx) / len(fx) if fx else 0,
                    "actual": archive.actual.at[pid, w] if pid in archive.actual.index else 0,
                })
        print(season, gw, len(rows), flush=True)
    return pd.DataFrame(rows)


def features(df):
    played = (df["gw"] - 1).clip(lower=1)
    mins = df["minutes"].clip(lower=1)
    per90 = lambda col: df[col] / mins * 90
    out = pd.DataFrame({
        "engine_xp": df["engine_xp"], "k": df["k"], "gw": df["gw"],
        "element_type": df["element_type"], "now_cost": df["now_cost"],
        "minutes_per_gw": df["minutes"] / played, "start_rate": df["starts"] / played,
        "recent_4_minutes": df["recent_4_minutes"],
        "points_per_gw": df["total_points"] / played,
        "points_per_start": df["total_points"] / df["starts"].clip(lower=1),
        "xg90": per90("expected_goals"), "xa90": per90("expected_assists"), "xgc90": per90("expected_goals_conceded"),
        "influence90": per90("influence"), "ict90": per90("ict_index"),
        "goals_minus_xg": (df["goals_scored"] - df["expected_goals"]) / played,
        "prior_points": df["prior_points"], "prior_minutes": df["prior_minutes"],
        "prior_points_per_start": df["prior_points"] / df["prior_starts"].clip(lower=1),
        "fixtures": df["fixtures"], "home": df["home"], "difficulty": df["difficulty"],
    })
    return out.astype(float)


def evaluate(seasons):
    from sklearn.ensemble import HistGradientBoostingRegressor

    data = pd.concat([pd.read_pickle(os.path.join(ROWS_DIR, f"rows_{s}.pkl")) for s in seasons], ignore_index=True)
    X, y = features(data), data["actual"].astype(float)
    playing = data["recent_4_minutes"] >= PLAYING_MINUTES
    data["ml_xp"] = np.nan
    for test in seasons:
        train = (data["season"] != test) & playing
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              categorical_features=[list(X.columns).index("element_type")],
                                              random_state=0)
        model.fit(X[train], y[train])
        held = data["season"] == test
        data.loc[held, "ml_xp"] = model.predict(X[held])
    data["blend_xp"] = 0.5 * data["engine_xp"] + 0.5 * data["ml_xp"]
    data.to_pickle(os.path.join(ROWS_DIR, "scored.pkl"))

    for test in seasons:
        df = data[data.season == test]
        e = metrics(df, "engine_xp")
        for col in ("ml_xp", "blend_xp"):
            m = metrics(df, col)
            print(test, col)
            for k in e:
                dc = np.array(m[k]["corr"]) - np.array(e[k]["corr"])
                print(f"  h{k}: corr {np.mean(e[k]['corr']):.3f} -> {np.mean(m[k]['corr']):.3f} "
                      f"({dc.mean():+.3f}±{1.96 * dc.std() / np.sqrt(len(dc)):.3f})  "
                      f"mae {np.mean(e[k]['mae']):.2f} -> {np.mean(m[k]['mae']):.2f}  "
                      f"top40 {np.mean(e[k]['top40']):.2f} -> {np.mean(m[k]['top40']):.2f}")


def metrics(df, col, horizons=(1, 5, 8)):
    res = {k: {"corr": [], "mae": [], "top40": []} for k in horizons}
    for gw, g in df[(df.gw >= 5) & (df.gw <= 31)].groupby("gw"):
        reg = g[g.recent_4_minutes >= 240]
        for k in horizons:
            s = reg[reg.k < k].groupby("id")[[col, "actual"]].sum()
            res[k]["corr"].append(np.corrcoef(s[col], s["actual"])[0, 1])
            res[k]["mae"].append((s[col] - s["actual"]).abs().mean() / k)
            res[k]["top40"].append(s.loc[s[col].nlargest(40).index, "actual"].mean() / k)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["build", "evaluate"])
    parser.add_argument("seasons", nargs="+")
    args = parser.parse_args()
    if args.action == "build":
        os.makedirs(ROWS_DIR, exist_ok=True)
        for season in args.seasons:
            build_rows(season).to_pickle(os.path.join(ROWS_DIR, f"rows_{season}.pkl"))
    else:
        evaluate(args.seasons)


if __name__ == "__main__":
    main()
