import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

import telemetry
from main import app, fpl_service
from telemetry_report import summarise


def log_lines(stdout: str):
    return [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]


class RequestLogTests(unittest.TestCase):
    def setUp(self):
        # Without `with`, the client skips startup, so nothing fetches FPL data.
        self.client = TestClient(app, raise_server_exceptions=False)

    def request(self, method, path, **kwargs):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            response = getattr(self.client, method)(path, **kwargs)
        lines = [l for l in log_lines(out.getvalue()) if l["kind"] == "request"]
        self.assertEqual(len(lines), 1, out.getvalue())
        return response, lines[0]

    def test_each_request_logs_its_route_status_duration_and_commit(self):
        response, line = self.request("get", "/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(line["endpoint"], "/health")
        self.assertEqual(line["status"], 200)
        self.assertEqual(line["severity"], "INFO")
        self.assertIn("duration_ms", line)
        self.assertEqual(line["commit"], telemetry.COMMIT)

    def test_a_manager_in_the_path_is_logged_as_a_hash_under_the_route_template(self):
        with mock.patch.object(fpl_service, "get_manager_chips", return_value={"available": []}), \
                mock.patch.object(fpl_service, "get_manager_entry", return_value={"name": "krAne"}):
            _, line = self.request("get", "/api/manager-chips/7706676")
        self.assertEqual(line["endpoint"], "/api/manager-chips/{manager_id}")
        self.assertEqual(line["manager"], telemetry.manager_hash(7706676))
        self.assertNotIn("7706676", json.dumps(line))

    def test_a_failed_request_logs_the_original_exception_type_without_the_manager_id(self):
        error = KeyError("https://fantasy.premierleague.com/api/entry/7706676/")
        with mock.patch.object(fpl_service, "get_manager_chips", side_effect=error), \
                mock.patch.object(fpl_service, "get_manager_entry", return_value={"name": "krAne"}):
            response, line = self.request("get", "/api/manager-chips/7706676")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(line["severity"], "ERROR")
        self.assertEqual(line["error_type"], "KeyError")
        self.assertNotIn("7706676", json.dumps(line))

    def test_a_manager_in_a_json_body_is_hashed_and_the_endpoint_still_reads_the_body(self):
        with mock.patch.object(fpl_service, "get_latest_data", side_effect=RuntimeError("offline")):
            response, line = self.request("post", "/api/chip-recommendations", json={"manager_id": 7706676})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "offline")
        self.assertEqual(line["manager"], telemetry.manager_hash(7706676))
        self.assertEqual(line["error_type"], "RuntimeError")

    def test_cors_preflight_for_the_telemetry_endpoint_is_allowed_and_not_logged(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            response = self.client.options("/api/telemetry", headers={
                "Origin": "https://fploptimize.netlify.app",
                "Access-Control-Request-Method": "POST",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://fploptimize.netlify.app")
        self.assertEqual(log_lines(out.getvalue()), [])


class TimedOutPlanner:
    """Runs out of time with chip weeks, then solves without them, as the planner does."""

    def __init__(self, **kwargs):
        pass

    def optimize_multi_period(self, plan_chips=True, **kwargs):
        if plan_chips:
            return self.optimize_multi_period(plan_chips=False, **kwargs)
        week = SimpleNamespace(gameweek=10, chip_used=None, transfers=SimpleNamespace(hits_taken=0, transfers_in=[{}]))
        return SimpleNamespace(status="Optimal", proven_optimal=False, optimization_time_seconds=15.04, total_expected_points=60.0,
                               gameweek_plans=[week], transfer_summary=[week.transfers])

    def to_dict(self, solution):
        return {"status": solution.status, "gameweek_plans": []}


class PlanRequestLogTests(unittest.TestCase):
    def test_a_plan_request_logs_its_settings_and_how_the_solve_went(self):
        data = {"static": {"elements": [], "teams": [], "events": [{"id": 10, "is_next": True}]}, "fixtures": []}
        client = TestClient(app, raise_server_exceptions=False)
        out = io.StringIO()
        with mock.patch.object(fpl_service, "get_latest_data", return_value=data), \
                mock.patch("main.MultiPeriodFPLOptimizer", TimedOutPlanner), \
                contextlib.redirect_stdout(out):
            response = client.post("/api/optimize/compare", json={"gameweeks": 8, "budget": 100})
        self.assertEqual(response.status_code, 200, response.text)
        line = [l for l in log_lines(out.getvalue()) if l["kind"] == "request"][0]
        self.assertEqual(line["endpoint"], "/api/optimize/compare")
        self.assertEqual((line["horizon"], line["plan_chips"], line["linked"]), (8, True, False))
        self.assertEqual((line["solves"], line["fallbacks"], line["resolves"]), (2, 1, 0))
        self.assertEqual(line["with_hits"], {"status": "Optimal", "proven_optimal": False, "solve_seconds": 15.04,
                                             "hits": 0, "deadline_transfers": 1})
        self.assertNotIn("manager", line)


class ClientEventTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def post(self, body):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            response = self.client.post("/api/telemetry", content=body, headers={
                "Content-Type": "text/plain", "Origin": "https://fploptimize.netlify.app",
            })
        return response, [l for l in log_lines(out.getvalue()) if l["kind"] == "client"]

    def test_known_events_are_logged_one_line_each_with_the_session(self):
        body = json.dumps({"session": "abc12345-def", "events": [
            {"event": "page_view", "props": {"tab": "planner", "mobile": True}, "t": 12.4},
            {"event": "plan_shown", "props": {"ms": 20500, "horizon": 8}, "t": 21000},
        ]})
        response, lines = self.post(body)
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://fploptimize.netlify.app")
        self.assertEqual([l["event"] for l in lines], ["page_view", "plan_shown"])
        self.assertEqual(lines[0]["session"], "abc12345-def")
        self.assertEqual(lines[0]["props"], {"tab": "planner", "mobile": True})
        self.assertEqual(lines[1]["t_ms"], 21000)

    def test_unknown_events_and_bad_props_are_dropped_and_long_numbers_scrubbed(self):
        body = json.dumps({"session": "abc12345", "events": [
            {"event": "drop_table", "props": {}},
            {"event": "error_shown", "props": {
                "message": "Not found: /api/entry/7706676/" + "x" * 500,
                "Bad Key": 1, "nested": {"a": 1}, "ratio": 1.5, "huge": float("inf"),
            }},
        ]})
        response, lines = self.post(body)
        self.assertEqual(response.status_code, 204)
        self.assertEqual(len(lines), 1)
        props = lines[0]["props"]
        self.assertEqual(set(props), {"message", "ratio"})
        self.assertNotIn("7706676", props["message"])
        self.assertLessEqual(len(props["message"]), 120)
        self.assertEqual(lines[0]["severity"], "WARNING")

    def test_a_body_over_the_size_limit_is_rejected_unread(self):
        body = json.dumps({"session": "abc12345", "events": [{"event": "page_view", "props": {"p": "x" * 9000}}]})
        response, lines = self.post(body)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(lines, [])

    def test_garbage_or_a_missing_session_is_rejected(self):
        for body in ["not json", "[]", json.dumps({"events": [{"event": "page_view"}]}),
                     json.dumps({"session": "abc12345", "events": [{"event": "nope"}]})]:
            response, lines = self.post(body)
            self.assertEqual(response.status_code, 400, body)
            self.assertEqual(lines, [])

    def test_only_the_first_twenty_events_of_a_batch_are_kept(self):
        body = json.dumps({"session": "abc12345", "events": [{"event": "tab_view"}] * 30})
        _, lines = self.post(body)
        self.assertEqual(len(lines), 20)


class ManagerHashTests(unittest.TestCase):
    def test_the_hash_is_stable_and_does_not_contain_the_id(self):
        self.assertEqual(telemetry.manager_hash(7706676), telemetry.manager_hash("7706676"))
        self.assertNotEqual(telemetry.manager_hash(7706676), telemetry.manager_hash(7706677))
        self.assertNotIn("7706676", telemetry.manager_hash(7706676))

    def test_a_different_salt_gives_a_different_hash(self):
        before = telemetry.manager_hash(7706676)
        with mock.patch.object(telemetry, "_SALT", "another salt"):
            self.assertNotEqual(telemetry.manager_hash(7706676), before)

    def test_no_manager_logs_nothing(self):
        self.assertIsNone(telemetry.manager_hash(None))
        self.assertIsNone(telemetry.manager_hash(0))


class FakePlanner:
    """Recurses like the planner: a re-solve around a Free Hit, then a fallback without chips."""

    def optimize_multi_period(self, plan_chips=True, free_hit_week=None, hold_override=None):
        if plan_chips and free_hit_week is None:
            return self.optimize_multi_period(plan_chips=True, free_hit_week=12, hold_override={})
        if plan_chips:
            return self.optimize_multi_period(plan_chips=False)
        return "plan"


class SolveCountTests(unittest.TestCase):
    def test_re_solves_and_fallbacks_are_counted_on_the_request_line(self):
        fields = {}
        token = telemetry._fields.set(fields)
        try:
            planner = FakePlanner()
            telemetry.count_solves(planner, plan_chips=True)
            self.assertEqual(planner.optimize_multi_period(plan_chips=True), "plan")
        finally:
            telemetry._fields.reset(token)
        self.assertEqual(fields, {"solves": 3, "resolves": 1, "fallbacks": 1})

    def test_a_plan_asked_for_without_chips_has_no_fallbacks(self):
        fields = {}
        token = telemetry._fields.set(fields)
        try:
            planner = FakePlanner()
            telemetry.count_solves(planner, plan_chips=False)
            planner.optimize_multi_period(plan_chips=False)
        finally:
            telemetry._fields.reset(token)
        self.assertEqual(fields, {"solves": 1, "resolves": 0, "fallbacks": 0})


class ReportTests(unittest.TestCase):
    def test_the_report_counts_latency_errors_solver_trouble_and_the_funnel(self):
        def req(endpoint, ms, status=200, **extra):
            return {"jsonPayload": {"kind": "request", "endpoint": endpoint, "duration_ms": ms, "status": status,
                                    "commit": "abc", **extra}}

        def client(session, event, **props):
            return {"jsonPayload": {"kind": "client", "session": session, "event": event, "props": props}}

        entries = [
            req("/health", 5), req("/health", 7),
            req("/api/optimize/compare", 20000, horizon=8, manager="h1", with_hits={"proven_optimal": True}),
            req("/api/optimize/compare", 75000, status=504, horizon=12, manager="h2", timed_out="with_hits"),
            req("/api/optimize/compare", 30000, horizon=12, fallbacks=1, with_hits={"proven_optimal": False}),
            client("s1", "page_view", mobile=True), client("s1", "manager_entered"),
            client("s1", "plan_requested"), client("s1", "plan_shown", ms=21000),
            client("s2", "page_view", mobile=False), client("s2", "manager_entered"),
            client("s2", "plan_requested"), client("s2", "error_shown", where="planner"),
        ]
        s = summarise(entries)
        compare = s["endpoints"]["/api/optimize/compare"]
        self.assertEqual((compare["requests"], compare["p50_ms"], compare["p95_ms"]), (3, 30000, 75000))
        self.assertEqual(compare["error_rate"], round(1 / 3, 3))
        self.assertEqual(s["solver"]["not_proven_optimal"], 1)
        self.assertEqual(s["solver"]["timed_out"], 1)
        self.assertEqual(s["solver"]["fallbacks"], 1)
        self.assertEqual(s["funnel"], {"page_view": 2, "manager_entered": 2, "plan_requested": 2, "plan_shown": 1})
        self.assertEqual(s["time_to_plan_ms"]["p50"], 21000)
        self.assertEqual(s["errors_shown"], {"planner": 1})
        self.assertEqual(s["mobile_share"], 0.5)
        self.assertEqual(s["managers"], 2)


if __name__ == "__main__":
    unittest.main()
