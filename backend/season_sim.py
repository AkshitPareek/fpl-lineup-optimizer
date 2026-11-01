"""
Replay a completed season with the multi-period optimizer and score actual points.

Each deadline: rebuild point-in-time player state, project with PointPredictor,
solve the plan, play its first week (XI, captain, bench order, transfers, hits),
score real points with FPL autosubs and vice-captain rules, roll the squad,
bank and free transfers forward. Chips (not the Wildcard) are played once per
half of the season by one of three policies: none, the chip advisor's rules
(`advisor`), or the plan placing them itself (`planner`).

Usage (from the backend dir):
  python season_sim.py --season 2025-26 --variant bench --chips planner
"""
import argparse
import json
import sys
import time

import pandas as pd

import point_predictor
point_predictor.UNDERSTAT_AVAILABLE = False  # current Understat totals would leak later weeks

import advanced_optimizer as ao
import predictor_eval as pe
from point_predictor import PointPredictor

VARIANTS = {
    "base":         dict(HORIZON_DECAY=1.0, BENCH_GK_WEIGHT=0.1, BENCH_WEIGHTS=(0.1, 0.1, 0.1)),
    "decay90":      dict(HORIZON_DECAY=0.9, BENCH_GK_WEIGHT=0.1, BENCH_WEIGHTS=(0.1, 0.1, 0.1)),
    "decay80":      dict(HORIZON_DECAY=0.8, BENCH_GK_WEIGHT=0.1, BENCH_WEIGHTS=(0.1, 0.1, 0.1)),
    "bench":        dict(HORIZON_DECAY=1.0, BENCH_GK_WEIGHT=0.03, BENCH_WEIGHTS=(0.25, 0.08, 0.03)),
    "decay90bench": dict(HORIZON_DECAY=0.9, BENCH_GK_WEIGHT=0.03, BENCH_WEIGHTS=(0.25, 0.08, 0.03)),
    "decay80bench": dict(HORIZON_DECAY=0.8, BENCH_GK_WEIGHT=0.03, BENCH_WEIGHTS=(0.25, 0.08, 0.03)),
}
STRENGTHS = ["strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"]
MIN_PLAYERS = {1: 1, 2: 3, 3: 2, 4: 1}


CHIPS = ["triple_captain", "bench_boost", "free_hit"]


def autosub_score(xi, bench, captain_id, vice_id, pts, mins, chip=None):
    """Actual points for a lineup under FPL autosub, vice-captain and chip rules."""
    pos = {p["id"]: p["element_type"] for p in xi + bench}
    lineup = [p["id"] for p in xi]
    subs = [p["id"] for p in bench]
    gk_sub = [j for j in subs if pos[j] == 1]
    out_subs = [j for j in subs if pos[j] != 1]
    for j in list(lineup):
        if mins.get(j, 0) > 0:
            continue
        if pos[j] == 1:
            if gk_sub and mins.get(gk_sub[0], 0) > 0:
                lineup[lineup.index(j)] = gk_sub.pop(0)
            continue
        for s in list(out_subs):
            if mins.get(s, 0) == 0:
                continue
            trial = [x for x in lineup if x != j] + [s]
            counts = {k: sum(1 for x in trial if pos[x] == k) for k in MIN_PLAYERS}
            if all(counts[k] >= MIN_PLAYERS[k] for k in MIN_PLAYERS):
                lineup[lineup.index(j)] = s
                out_subs.remove(s)
                break
    if chip == "bench_boost":
        lineup = [p["id"] for p in xi + bench]
    total = sum(pts.get(j, 0) for j in lineup)
    extra = 2 if chip == "triple_captain" else 1
    if captain_id in lineup and mins.get(captain_id, 0) > 0:
        total += extra * pts.get(captain_id, 0)
    elif vice_id in lineup and mins.get(vice_id, 0) > 0:
        total += extra * pts.get(vice_id, 0)
    return total


def run(season, variant, start, end, horizon, chips="none"):
    for k, v in VARIANTS[variant].items():
        setattr(ao.MultiPeriodFPLOptimizer, k, v)
    ao.MultiPeriodFPLOptimizer.generate_transfer_alternatives = lambda self, **kw: []
    if chips != "advisor":
        ao.ChipAdvisor.analyze_chip_opportunities_for_horizon = lambda self, **kw: {}

    archive = pe.SeasonArchive(season)
    prior = pe.SeasonArchive(pe.previous_season(season)).season_totals_by_code()
    teams = archive.teams.copy()
    teams[STRENGTHS] = 0  # like 2026/27, so the live fixture-difficulty fallback is what runs
    minutes = archive.gws.groupby(["element", "round"])["minutes"].sum()

    squad, bank, ft = [], 0.0, 1
    total, hits_total, log = 0.0, 0, []
    used = []  # chips played this half; the Wildcard is never played

    for gw in range(start, end + 1):
        t0 = time.time()
        state = archive.state_before(gw, prior)
        state["points_per_game"] = state["points_per_game"].astype(float)
        k = min(horizon, 39 - gw)
        preds = PointPredictor(state, teams, archive.fixtures, gw, use_understat=False).predict_all_players(gameweeks=k)
        opt = ao.MultiPeriodFPLOptimizer(state, teams, archive.fixtures, gw, predictions_df=preds)
        price = dict(zip(state["id"], state["now_cost"] / 10.0))
        squad = [j for j in squad if j in price]
        budget = 100.0 if not squad else sum(price[j] for j in squad) + bank
        if gw == ao.MultiPeriodFPLOptimizer.FIRST_HALF_END + 1:
            used = []
        spent = ["wildcard"] + (CHIPS if chips == "none" else used)
        args = dict(budget=budget, gameweeks=k, current_squad_ids=squad, banked_transfers=ft,
                    chips_used=spent, max_hits=4, selling_prices={j: price[j] for j in squad})
        sol = opt.optimize_multi_period(**args, plan_chips=chips == "planner" and bool(squad))
        advice = sol.chip_suggestion if chips == "advisor" else None
        if advice and advice.get("recommended_gameweek") == gw and advice.get("chip") in CHIPS and squad:
            sol = opt.optimize_multi_period(**args, chip_to_use=(advice["chip"], gw))
        if not sol.gameweek_plans:
            print(f"GW{gw}: solver status {sol.status}", file=sys.stderr)
            break
        plan = sol.gameweek_plans[0]
        pts = archive.actual[gw].to_dict() if gw in archive.actual.columns else {}
        mins = minutes.xs(gw, level="round").to_dict() if gw in minutes.index.get_level_values("round") else {}
        hits = plan.transfers.hits_taken if plan.transfers else 0
        chip = opt._chip_key(plan.chip_used.value) if plan.chip_used else None
        if chip:
            used.append(chip)
        week = autosub_score(plan.starting_xi, plan.bench, plan.captain_id, plan.vice_captain_id, pts, mins, chip) - 4 * hits
        # What Triple Captain or Bench Boost added that week; a Free Hit's gain has no counterpart to score.
        chip_points = None
        if chip in ("triple_captain", "bench_boost"):
            chip_points = week + 4 * hits - autosub_score(plan.starting_xi, plan.bench, plan.captain_id,
                                                          plan.vice_captain_id, pts, mins)
        total += week
        hits_total += hits
        spent_ft = plan.transfers.free_transfers_used if (plan.transfers and squad) else 0
        if chip != "free_hit":  # a Free Hit squad and bank go back afterwards; saved transfers are kept
            ft = 1 if not squad else min(5, max(0, ft - spent_ft) + 1)
            squad = [p["id"] for p in plan.starting_xi + plan.bench]
            bank = plan.bank
        log.append(dict(gw=gw, points=week, hits=hits, chip=chip, chip_points=chip_points, xp=round(plan.expected_points, 1),
                        secs=round(time.time() - t0, 1)))
        print(f"{variant}/{chips} GW{gw}: {week:5.1f} pts (hits {hits}{', ' + chip if chip else ''}) "
              f"total {total:6.1f}  [{time.time() - t0:.1f}s]", flush=True)
    return dict(season=season, variant=variant, chips=chips, total=total, hits=hits_total, weeks=log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--variant", default="base", choices=list(VARIANTS))
    ap.add_argument("--start", type=int, default=5)
    ap.add_argument("--end", type=int, default=38)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--chips", default="none", choices=["none", "advisor", "planner"])
    ap.add_argument("--out")
    a = ap.parse_args()
    res = run(a.season, a.variant, a.start, a.end, a.horizon, a.chips)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("season", "variant", "chips", "total", "hits")}))
