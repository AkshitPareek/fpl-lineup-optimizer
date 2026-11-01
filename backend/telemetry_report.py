"""
Summarise the last N days of telemetry from Cloud Logging.

    python telemetry_report.py --days 7
    python telemetry_report.py --file entries.json   # output of `gcloud logging read --format=json`

Needs gcloud logged into the project (or --file). Standard library only.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict

SERVICE = "fpl-optimizer-api"
FUNNEL = ["page_view", "manager_entered", "plan_requested", "plan_shown"]


def read_entries(days: int, project: str = None, service: str = SERVICE):
    log_filter = (
        f'resource.type="cloud_run_revision" AND resource.labels.service_name="{service}" '
        f'AND (jsonPayload.kind="request" OR jsonPayload.kind="client")'
    )
    cmd = ["gcloud", "logging", "read", log_filter, f"--freshness={days}d", "--format=json", "--limit=100000"]
    if project:
        cmd.append(f"--project={project}")
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout or "[]")


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summarise(entries):
    """Turn log entries (or bare payloads) into the numbers the report prints."""
    payloads = [e.get("jsonPayload", e) for e in entries]
    requests = [p for p in payloads if p.get("kind") == "request"]
    client = [p for p in payloads if p.get("kind") == "client"]

    endpoints = defaultdict(list)
    for r in requests:
        endpoints[r.get("endpoint", "?")].append(r)
    by_endpoint = {}
    for name, rows in endpoints.items():
        durations = [r["duration_ms"] for r in rows if "duration_ms" in r]
        by_endpoint[name] = {
            "requests": len(rows),
            "p50_ms": percentile(durations, 0.5),
            "p95_ms": percentile(durations, 0.95),
            "error_rate": round(sum(r.get("status", 0) >= 500 for r in rows) / len(rows), 3),
        }

    plans = [r for r in requests if "horizon" in r]
    first_solves = [r.get("with_hits") or r.get("plan") or {} for r in plans]
    solver = {
        "plans": len(plans),
        "not_proven_optimal": sum(s.get("proven_optimal") is False for s in first_solves),
        "timed_out": sum("timed_out" in r for r in plans),
        "fallbacks": sum(r.get("fallbacks", 0) for r in plans),
        "resolves": sum(r.get("resolves", 0) for r in plans),
        "horizons": dict(sorted(Counter(r["horizon"] for r in plans).items())),
    }

    sessions = defaultdict(set)
    for c in client:
        sessions[c.get("session")].add(c.get("event"))
    funnel = {step: sum(step in events for events in sessions.values()) for step in FUNNEL}
    plan_ms = [c["props"]["ms"] for c in client
               if c.get("event") == "plan_shown" and isinstance(c.get("props", {}).get("ms"), (int, float))]
    errors_shown = Counter(c.get("props", {}).get("where", "?") for c in client if c.get("event") == "error_shown")
    mobile = [c.get("props", {}).get("mobile") for c in client if c.get("event") == "page_view"]

    return {
        "endpoints": dict(sorted(by_endpoint.items(), key=lambda kv: -kv[1]["requests"])),
        "solver": solver,
        "sessions": len(sessions),
        "funnel": funnel,
        "time_to_plan_ms": {"p50": percentile(plan_ms, 0.5), "p95": percentile(plan_ms, 0.95)},
        "errors_shown": dict(errors_shown),
        "mobile_share": round(sum(bool(m) for m in mobile) / len(mobile), 2) if mobile else None,
        "managers": len({r["manager"] for r in requests if r.get("manager")}),
        "commits": dict(Counter(r.get("commit") for r in requests)),
    }


def print_report(s):
    print(f"{'endpoint':42} {'reqs':>6} {'p50 s':>7} {'p95 s':>7} {'5xx':>6}")
    for name, e in s["endpoints"].items():
        p50 = f"{e['p50_ms'] / 1000:.1f}" if e["p50_ms"] is not None else "-"
        p95 = f"{e['p95_ms'] / 1000:.1f}" if e["p95_ms"] is not None else "-"
        print(f"{name:42} {e['requests']:>6} {p50:>7} {p95:>7} {e['error_rate']:>6.1%}")
    sv = s["solver"]
    print(f"\nPlans: {sv['plans']}  not proven optimal: {sv['not_proven_optimal']}  "
          f"timed out (504 or partial): {sv['timed_out']}  fallbacks: {sv['fallbacks']}  re-solves: {sv['resolves']}")
    print(f"Horizons asked for: {sv['horizons']}")
    print(f"\nSessions: {s['sessions']}  distinct managers: {s['managers']}  mobile share: {s['mobile_share']}")
    print("Funnel (sessions reaching each step): " + " -> ".join(f"{k} {v}" for k, v in s["funnel"].items()))
    t = s["time_to_plan_ms"]
    if t["p50"] is not None:
        print(f"Time to plan in the browser: p50 {t['p50'] / 1000:.1f}s, p95 {t['p95'] / 1000:.1f}s")
    print(f"Errors shown to users, by view: {s['errors_shown']}")
    print(f"Requests by deployed commit: {s['commits']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--project")
    parser.add_argument("--service", default=SERVICE)
    parser.add_argument("--file", help="read entries from a JSON file instead of gcloud")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            entries = json.load(f)
    else:
        entries = read_entries(args.days, args.project, args.service)
    summary = summarise(entries)
    if args.json:
        json.dump(summary, sys.stdout, indent=2)
        print()
    else:
        print_report(summary)


if __name__ == "__main__":
    main()
