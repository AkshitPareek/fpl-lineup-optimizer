"""
Weekly scorecard: what the live planner advised before each deadline, and what that advice scored.

Before a deadline, `record` asks the live API for the manager's plan and saves the
first week (XI, bench order, captain, transfers, chip). Once the gameweek is over,
`score` plays that lineup against real points with FPL autosubs and compares it
with the manager's own score, the average manager and the week's best.

The advice is scored from the squad the manager actually held that week, so it
answers: what would following this week's advice have scored?

Usage (from the backend dir):
    python scorecard.py record --manager 7706676 [--within-hours 4]
    python scorecard.py score --manager 7706676
"""
import argparse
import json
import os
from datetime import datetime, timezone

import requests

API_URL = os.getenv("PLANNER_API_URL", "https://fpl-optimizer-api-1099379580588.asia-south1.run.app")
FPL_URL = "https://fantasy.premierleague.com/api"
RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scorecard")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
MIN_PLAYERS = {1: 1, 2: 3, 3: 2, 4: 1}


def fpl(path):
    response = requests.get(f"{FPL_URL}/{path}", headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def record_path(manager_id, gameweek):
    return os.path.join(RECORDS_DIR, str(manager_id), f"gw{gameweek:02d}.json")


def record(manager_id, gameweeks=8, within_hours=None):
    events = fpl("bootstrap-static/")["events"]
    upcoming = next(e for e in events if e["is_next"])
    path = record_path(manager_id, upcoming["id"])
    if os.path.exists(path):
        print(f"GW{upcoming['id']} already recorded: {path}")
        return
    deadline = datetime.fromisoformat(upcoming["deadline_time"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) >= deadline:
        raise SystemExit(f"GW{upcoming['id']} deadline has passed; advice now would not be honest")
    hours_left = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
    if within_hours is not None and hours_left > within_hours:
        print(f"GW{upcoming['id']} deadline is {hours_left:.1f}h away; recording waits for team news")
        return

    account = requests.get(f"{API_URL}/api/manager-chips/{manager_id}", timeout=60).json()
    body = {"manager_id": manager_id, "gameweeks": gameweeks,
            "banked_transfers": account.get("banked_transfers", 1)}
    plan = requests.post(f"{API_URL}/api/optimize/multi-period", json=body, timeout=300)
    plan.raise_for_status()
    plan = plan.json()
    week = plan["gameweek_plans"][0]
    if week["gameweek"] != upcoming["id"]:
        raise SystemExit(f"Planner answered for GW{week['gameweek']}, expected GW{upcoming['id']}")
    version = requests.get(f"{API_URL}/health", timeout=30).json().get("version")

    def slim(players):
        return [{"id": p["id"], "name": p["web_name"], "element_type": p["element_type"],
                 "xp": round(p["expected_points"], 2)} for p in players]

    transfers = week.get("transfers") or {}
    entry = {
        "manager_id": manager_id,
        "gameweek": week["gameweek"],
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deadline": upcoming["deadline_time"],
        "api_version": version,
        "expected_points": round(week["expected_points"], 2),
        "starting_xi": slim(week["starting_xi"]),
        "bench": slim(week["bench"]),
        "captain_id": week["captain_id"],
        "vice_captain_id": week["vice_captain_id"],
        "chip": week.get("chip_used"),
        "transfers_in": [t["name"] for t in transfers.get("in", [])],
        "transfers_out": [t["name"] for t in transfers.get("out", [])],
        "hits": transfers.get("hits", 0) or 0,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(entry, f, indent=1)
    print(f"Recorded GW{entry['gameweek']}: {entry['transfers_out']} -> {entry['transfers_in']}, "
          f"captain {next(p['name'] for p in entry['starting_xi'] if p['id'] == entry['captain_id'])}, "
          f"{entry['expected_points']} xP")


def lineup_points(entry, points, minutes):
    """Real points for a recorded lineup under FPL autosub, vice-captain and chip rules."""
    xi, bench = entry["starting_xi"], entry["bench"]
    pos = {p["id"]: p["element_type"] for p in xi + bench}
    if entry["chip"] == "bench_boost":
        lineup = [p["id"] for p in xi + bench]
    else:
        lineup = [p["id"] for p in xi]
        keeper_subs = [p["id"] for p in bench if p["element_type"] == 1]
        outfield_subs = [p["id"] for p in bench if p["element_type"] != 1]
        for j in [p["id"] for p in xi]:
            if minutes.get(j, 0) > 0:
                continue
            if pos[j] == 1:
                if keeper_subs and minutes.get(keeper_subs[0], 0) > 0:
                    lineup[lineup.index(j)] = keeper_subs.pop(0)
                continue
            for s in list(outfield_subs):
                if minutes.get(s, 0) == 0:
                    continue
                trial = [x for x in lineup if x != j] + [s]
                if all(sum(1 for x in trial if pos[x] == k) >= n for k, n in MIN_PLAYERS.items()):
                    lineup[lineup.index(j)] = s
                    outfield_subs.remove(s)
                    break
    total = sum(points.get(j, 0) for j in lineup)
    multiplier = 2 if entry["chip"] == "triple_captain" else 1
    for armband in (entry["captain_id"], entry["vice_captain_id"]):
        if armband in lineup and minutes.get(armband, 0) > 0:
            total += multiplier * points.get(armband, 0)
            break
    return total - 4 * entry["hits"]


def score(manager_id):
    folder = os.path.join(RECORDS_DIR, str(manager_id))
    if not os.path.isdir(folder):
        raise SystemExit(f"No records for manager {manager_id}")
    events = {e["id"]: e for e in fpl("bootstrap-static/")["events"]}
    history = {h["event"]: h for h in fpl(f"entry/{manager_id}/history/")["current"]}

    rows = []
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name)) as f:
            entry = json.load(f)
        gw = entry["gameweek"]
        if not events.get(gw, {}).get("finished"):
            rows.append((gw, None, None, None, None))
            continue
        live = fpl(f"event/{gw}/live/")["elements"]
        points = {e["id"]: e["stats"]["total_points"] for e in live}
        minutes = {e["id"]: e["stats"]["minutes"] for e in live}
        mine = history.get(gw)
        you = mine["points"] - mine["event_transfers_cost"] if mine else None
        rows.append((gw, lineup_points(entry, points, minutes), you,
                     events[gw]["average_entry_score"], events[gw]["highest_score"]))

    print(f"{'GW':>3} {'advice':>7} {'you':>5} {'average':>8} {'best':>5}")
    totals = [0, 0, 0]
    for gw, advice, you, average, best in rows:
        if advice is None:
            print(f"{gw:>3}   (not finished)")
            continue
        totals = [totals[0] + advice, totals[1] + (you or 0), totals[2] + average]
        print(f"{gw:>3} {advice:>7} {you if you is not None else '-':>5} {average:>8} {best:>5}")
    print(f"{'all':>3} {totals[0]:>7} {totals[1]:>5} {totals[2]:>8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["record", "score"])
    parser.add_argument("--manager", type=int, required=True)
    parser.add_argument("--gameweeks", type=int, default=8)
    parser.add_argument("--within-hours", type=float, help="Only record once the deadline is this close")
    args = parser.parse_args()
    if args.action == "record":
        record(args.manager, args.gameweeks, args.within_hours)
    else:
        score(args.manager)


if __name__ == "__main__":
    main()
