"""
Structured telemetry: one JSON line on stdout per API request and per frontend event.

Cloud Run sends stdout to Cloud Logging, which parses a JSON line into jsonPayload,
so these lines can be filtered and summarised without another service. See
TELEMETRY.md for the queries and telemetry_report.py for a summary.

Manager IDs are public but still point at one person's team, so only a salted hash
is logged. The salt comes from TELEMETRY_SALT; without it each process makes its
own, and hashes then only link requests served by the same instance.
"""

import contextvars
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import time
from typing import Any, Dict, List, Optional

COMMIT = os.getenv("GIT_COMMIT", "unknown")
# Set by Cloud Run; tells two deploys of the same commit apart.
REVISION = os.getenv("K_REVISION")
_SALT = os.getenv("TELEMETRY_SALT") or secrets.token_hex(16)

# Fields the request's handlers add to its log line. Holds a dict per request.
_fields: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("telemetry_fields", default=None)

# Long digit runs are most likely manager IDs (in FPL URLs inside error messages).
_DIGITS = re.compile(r"\d{5,}")


def manager_hash(manager_id) -> Optional[str]:
    """A stable pseudonym for a manager ID; the raw ID is never logged."""
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return None
    if manager_id <= 0:
        return None
    return hmac.new(_SALT.encode(), str(manager_id).encode(), hashlib.sha256).hexdigest()[:16]


def scrub(text, limit: int = 200) -> str:
    return _DIGITS.sub("#", str(text))[:limit]


def emit(kind: str, message: str, severity: str = "INFO", **fields) -> None:
    """Print one log line. `severity` and `message` are the keys Cloud Logging reads."""
    line = {"severity": severity, "message": message, "kind": kind, "commit": COMMIT}
    if REVISION:
        line["revision"] = REVISION
    line.update({k: v for k, v in fields.items() if v is not None})
    print(json.dumps(line, default=str, separators=(",", ":")), flush=True)


def note(**fields) -> None:
    """Add fields to the current request's log line. Does nothing outside a request."""
    current = _fields.get()
    if current is not None:
        current.update({k: v for k, v in fields.items() if v is not None})


def note_error(exc: BaseException) -> None:
    """Record why a request failed. Handlers re-raise as HTTPException, so prefer the cause."""
    cause = exc.__cause__ or exc.__context__ or exc
    detail = getattr(exc, "detail", None) or str(cause)
    note(error_type=type(cause).__name__, error=scrub(detail))


def note_plan_request(request) -> None:
    """The planner settings of a /api/optimize/multi-period or /compare request."""
    note(
        horizon=request.gameweeks,
        plan_chips=request.plan_chips,
        strategy=request.strategy,
        robust=request.robust,
        forced_chip=request.chip_to_use[0] if request.chip_to_use else None,
        linked=bool(request.manager_id),
    )


def plan_summary(solution) -> Dict[str, Any]:
    """What a planner run produced, for the request's log line."""
    summary = {
        "status": getattr(solution, "status", None),
        "proven_optimal": getattr(solution, "proven_optimal", None),
        # Time of the last solve; re-solves and fallbacks each restart this clock.
        "solve_seconds": round(getattr(solution, "optimization_time_seconds", 0) or 0, 2),
    }
    plans = getattr(solution, "gameweek_plans", None) or []
    chips = {getattr(p.chip_used, "value", p.chip_used): p.gameweek for p in plans if p.chip_used}
    if chips:
        summary["chips"] = chips
    if plans:
        summary["hits"] = sum(p.transfers.hits_taken for p in plans if p.transfers)
        summary["deadline_transfers"] = len(plans[0].transfers.transfers_in) if plans[0].transfers else 0
    return summary


def count_solves(optimizer, plan_chips: bool) -> None:
    """
    Count the solves behind a plan on this request's log line.

    The planner re-solves around a Free Hit or when its chip weeks change, and falls
    back to a plan without chips when one with them is not solved in time. Those
    calls go through self.optimize_multi_period, so wrapping it on the instance sees
    them without the planner knowing about telemetry.
    """
    fields = _fields.get()
    if fields is None:
        return
    for key in ("solves", "resolves", "fallbacks"):
        fields.setdefault(key, 0)
    solve = optimizer.optimize_multi_period

    def counted(*args, **kwargs):
        # The planner runs in an executor thread, so update the dict captured above.
        fields["solves"] += 1
        if kwargs.get("hold_override") is not None or kwargs.get("free_hit_week"):
            fields["resolves"] += 1
        elif plan_chips and kwargs.get("plan_chips") is False:
            fields["fallbacks"] += 1
        return solve(*args, **kwargs)

    optimizer.optimize_multi_period = counted


def _manager_from_body(body: bytes):
    if not body or len(body) > 65536:
        return None
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    return payload.get("manager_id") if isinstance(payload, dict) else None


async def log_request(request, call_next):
    """HTTP middleware: time the request and log one line when it is done."""
    if request.method == "OPTIONS":
        return await call_next(request)
    fields: Dict[str, Any] = {}
    token = _fields.set(fields)
    start = time.perf_counter()
    status = 500
    manager = None
    try:
        if request.method == "POST" and request.url.path != "/api/telemetry":
            # Starlette caches the body, so the endpoint can still read it.
            manager = _manager_from_body(await request.body())
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception as exc:
        # Unhandled errors reach the app's 500 handler outside this middleware.
        fields.setdefault("error_type", type(exc).__name__)
        fields.setdefault("error", scrub(exc))
        raise
    finally:
        _fields.reset(token)
        # The router writes the matched route into the scope, so the endpoint is
        # its template (/api/manager/{manager_id}) rather than the raw path.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", None) or "unmatched"
        manager = request.scope.get("path_params", {}).get("manager_id") or manager
        duration = time.perf_counter() - start
        severity = "ERROR" if status >= 500 else "WARNING" if status >= 400 else "INFO"
        emit(
            "request",
            f"{request.method} {endpoint} {status} {duration:.2f}s",
            severity=severity,
            method=request.method,
            endpoint=endpoint,
            status=status,
            duration_ms=round(duration * 1000),
            manager=manager_hash(manager),
            **fields,
        )


# Frontend events. Anything else is dropped, so a client cannot fill the logs with noise.
CLIENT_EVENTS = {
    "page_view", "tab_view", "manager_entered", "manager_loaded",
    "plan_requested", "plan_shown", "error_shown",
    "strategy_switched", "gameweek_viewed", "view_toggled", "transfer_why_opened",
    "chip_selected", "chip_suggestion_applied",
    "dream_team_requested", "backtest_requested", "backtest_shown",
}
MAX_BODY_BYTES = 8192
MAX_EVENTS = 20
MAX_PROPS = 16
_SESSION = re.compile(r"^[A-Za-z0-9-]{8,64}$")
_PROP_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class TelemetryRejected(ValueError):
    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status


def _clean_props(props) -> Dict[str, Any]:
    if not isinstance(props, dict):
        return {}
    clean = {}
    for key, value in list(props.items())[:MAX_PROPS]:
        if not isinstance(key, str) or not _PROP_KEY.match(key):
            continue
        if isinstance(value, bool) or value is None:
            clean[key] = value
        elif isinstance(value, (int, float)):
            if math.isfinite(value):
                clean[key] = value
        elif isinstance(value, str):
            clean[key] = scrub(value, 120)
    return clean


def parse_client_events(body: bytes) -> List[Dict[str, Any]]:
    """Validate a frontend batch: {"session": id, "events": [{"event", "props", "t"}]}."""
    if len(body) > MAX_BODY_BYTES:
        raise TelemetryRejected(413, "too large")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise TelemetryRejected(400, "not JSON")
    if not isinstance(payload, dict):
        raise TelemetryRejected(400, "not an object")
    session = payload.get("session")
    if not isinstance(session, str) or not _SESSION.match(session):
        raise TelemetryRejected(400, "bad session")
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise TelemetryRejected(400, "no events")

    parsed = []
    for raw in events[:MAX_EVENTS]:
        if not isinstance(raw, dict) or raw.get("event") not in CLIENT_EVENTS:
            continue
        t = raw.get("t")
        parsed.append({
            "event": raw["event"],
            "session": session,
            # Milliseconds since the page loaded, to order a session's events.
            "t_ms": round(t) if isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t >= 0 else None,
            "props": _clean_props(raw.get("props")),
        })
    if not parsed:
        raise TelemetryRejected(400, "no known events")
    return parsed


def log_client_events(events: List[Dict[str, Any]]) -> None:
    for e in events:
        severity = "WARNING" if e["event"] == "error_shown" else "INFO"
        emit("client", f"client {e['event']}", severity=severity, **e)
