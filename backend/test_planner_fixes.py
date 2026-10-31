import json
import os
import tempfile
import time
import unittest
from unittest import mock

import pandas as pd

from fpl_service import FPLService, banked_transfers_for_next_deadline, manager_squad_state, recent_minutes, selling_price
from main import _comparison_entry


class RecentMinutesTests(unittest.TestCase):
    def test_last_four_weeks_his_club_played(self):
        def live(rows):
            return {"elements": [{"id": pid, "stats": {"minutes": m}, "explain": [] if m is None else [{}]}
                                 for pid, m in rows]}
        # Newest first. Player 1's club blanked in the second week; player 2 is new.
        lives = [live([(1, 0), (2, 90)]), live([(1, None)]), live([(1, 0)]), live([(1, 90)]),
                 live([(1, 90)]), live([(1, 90)])]
        elements = [{"id": 1}, {"id": 2}, {"id": 3}]
        recent_minutes(elements, lives)
        self.assertEqual(elements[0]["recent_4_minutes"], 0 + 0 + 90 + 90)
        self.assertEqual(elements[1]["recent_4_minutes"], 90)
        self.assertNotIn("recent_4_minutes", elements[2])


class BankedTransferTests(unittest.TestCase):
    def test_accrues_unused_transfers_and_spends_them(self):
        events = [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 0},
            {"event": 3, "event_transfers": 0},
            {"event": 4, "event_transfers": 2},
            {"event": 5, "event_transfers": 2},
        ]
        # GW1 grants nothing: 1 for GW2, 2 for GW3, 3 for GW4, then 2 for GW5 and 1 for GW6.
        self.assertEqual(banked_transfers_for_next_deadline(events), 1)

    def test_wildcard_keeps_saved_transfers_without_adding_one(self):
        events = [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 0},
            {"event": 3, "event_transfers": 0},
            {"event": 4, "event_transfers": 8},
        ]
        chips = [{"event": 4, "name": "wildcard"}]
        # Three saved going into the GW4 wildcard are still three for GW5.
        self.assertEqual(banked_transfers_for_next_deadline(events, chips), 3)

    def test_hits_leave_one_for_the_next_week(self):
        events = [{"event": 1, "event_transfers": 0}, {"event": 2, "event_transfers": 3}]
        self.assertEqual(banked_transfers_for_next_deadline(events), 1)

    def test_bank_is_capped_at_five(self):
        events = [{"event": gw, "event_transfers": 0} for gw in range(1, 10)]
        self.assertEqual(banked_transfers_for_next_deadline(events), 5)

    def test_late_joiner_starts_with_one_after_the_first_deadline(self):
        self.assertEqual(banked_transfers_for_next_deadline([{"event": 9, "event_transfers": 0}]), 1)

    def test_empty_history_starts_with_one(self):
        self.assertEqual(banked_transfers_for_next_deadline([]), 1)


class SellingPriceTests(unittest.TestCase):
    def test_keeps_half_of_a_rise_rounded_down(self):
        self.assertEqual(selling_price(75, 78), 76)
        self.assertEqual(selling_price(48, 51), 49)
        self.assertEqual(selling_price(57, 58), 57)

    def test_passes_on_a_fall_in_full(self):
        self.assertEqual(selling_price(62, 60), 60)


def elements(*rows):
    return [{"id": pid, "now_cost": now, "cost_change_start": change} for pid, now, change in rows]


class ManagerSquadStateTests(unittest.TestCase):
    def test_selling_prices_come_from_the_latest_purchase(self):
        history = {"current": [{"event": 1, "event_transfers": 0, "bank": 5},
                               {"event": 2, "event_transfers": 1, "bank": 3}], "chips": []}
        transfers = [
            {"event": 2, "element_in": 3, "element_in_cost": 48, "element_out": 9, "element_out_cost": 50,
             "time": "2026-08-29T10:00:00Z"},
        ]
        picks = {2: {"picks": [{"element": 1}, {"element": 2}, {"element": 3}]}}
        state = manager_squad_state(history, transfers, picks.get,
                                    elements((1, 78, 3), (2, 60, -2), (3, 51, 1)))
        # 1 was bought at the 7.5 start price, 2 has fallen, 3 was bought at 4.8.
        self.assertEqual(state["selling_prices"], {1: 7.6, 2: 6.0, 3: 4.9})
        self.assertEqual(state["bank"], 0.3)
        self.assertEqual(state["free_transfers"], 1)

    def test_transfers_already_made_for_the_next_deadline_are_applied(self):
        history = {"current": [{"event": 1, "event_transfers": 0, "bank": 10},
                               {"event": 2, "event_transfers": 0, "bank": 10}], "chips": []}
        transfers = [{"event": 3, "element_in": 4, "element_in_cost": 55, "element_out": 2, "element_out_cost": 60,
                      "time": "2026-09-01T10:00:00Z"}]
        picks = {2: {"picks": [{"element": 1}, {"element": 2}]}}
        state = manager_squad_state(history, transfers, picks.get, elements((1, 50, 0), (2, 60, 0), (4, 55, 0)))
        self.assertEqual(state["squad"], [1, 4])
        self.assertEqual(state["bank"], 1.5)
        self.assertEqual(state["free_transfers"], 1)
        self.assertEqual(state["selling_prices"][4], 5.5)

    def test_free_hit_squad_reverts(self):
        history = {"current": [{"event": 1, "event_transfers": 0, "bank": 0},
                               {"event": 2, "event_transfers": 0, "bank": 4},
                               {"event": 3, "event_transfers": 0, "bank": 20}],
                   "chips": [{"event": 3, "name": "freehit"}]}
        transfers = [{"event": 3, "element_in": 7, "element_in_cost": 40, "element_out": 1, "element_out_cost": 60,
                      "time": "2026-09-05T10:00:00Z"}]
        picks = {2: {"picks": [{"element": 1}]}, 3: {"picks": [{"element": 7}]}}
        state = manager_squad_state(history, transfers, picks.get, elements((1, 60, 0), (7, 40, 0)))
        self.assertEqual(state["squad"], [1])
        self.assertEqual(state["bank"], 0.4)
        # One for GW2, two for GW3, kept through the Free Hit.
        self.assertEqual(state["free_transfers"], 2)

    def test_late_joiner_squad_uses_the_price_paid(self):
        history = {"current": [{"event": 5, "event_transfers": 0, "bank": 0}], "chips": []}
        picks = {5: {"picks": [{"element": 1}]}}
        state = manager_squad_state(history, [], picks.get, elements((1, 64, 4)), initial_price=lambda pid: 62)
        self.assertEqual(state["selling_prices"], {1: 6.3})


class ManagerBankTests(unittest.TestCase):
    def test_reports_the_latest_bank_in_millions(self):
        service = FPLService()
        history = {"current": [{"event": 5, "event_transfers": 1, "bank": 12},
                               {"event": 6, "event_transfers": 0, "bank": 3}], "chips": []}
        with mock.patch.object(service, "_get_json", return_value=history), \
                mock.patch.object(service, "get_manager_squad_state",
                                  return_value={"bank": 0.3, "free_transfers": 2, "pending_transfers": 0}):
            chips = service.get_manager_chips(1)
        self.assertAlmostEqual(chips["bank"], 0.3)
        self.assertEqual(chips["banked_transfers"], 2)
        self.assertTrue(chips["synced"])

    def test_a_forbidden_response_is_retried(self):
        service = FPLService()
        calls = []

        class Response:
            def __init__(self, code, body):
                self.status_code = code
                self._body = body

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise __import__("requests").HTTPError(str(self.status_code))

            def json(self):
                return self._body

        def get(url, headers=None, timeout=None):
            calls.append(url)
            body = {"current": [], "chips": []}
            return Response(403 if len(calls) == 1 else 200, body)

        with mock.patch("fpl_service.requests.get", get), mock.patch("fpl_service.time.sleep"):
            chips = service.get_manager_chips(1)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(chips["synced"])

    def test_a_failed_account_read_is_not_treated_as_loaded(self):
        service = FPLService()
        with mock.patch.object(service, "_get_json", side_effect=__import__("requests").RequestException("nope")):
            chips = service.get_manager_chips(1)
        self.assertFalse(chips["synced"])
        self.assertNotIn("bank", chips)

    def test_unknown_bank_is_left_to_the_user(self):
        service = FPLService()
        with mock.patch.object(service, "_get_json", return_value={"current": [], "chips": []}):
            self.assertIsNone(service.get_manager_chips(1)["bank"])


class ManagerChipTests(unittest.TestCase):
    def chips(self, last_gw, played):
        service = FPLService()
        history = {"current": [{"event": gw, "event_transfers": 0, "bank": 0} for gw in range(1, last_gw + 1)],
                   "chips": [{"name": name, "event": gw} for name, gw in played]}
        with mock.patch.object(service, "_get_json", return_value=history), \
                mock.patch.object(service, "get_manager_squad_state", side_effect=KeyError("picks")):
            return service.get_manager_chips(1)

    def test_played_chips_are_not_offered_again(self):
        chips = self.chips(8, [("wildcard", 6), ("3xc", 7)])
        self.assertEqual(chips["available"], ["free_hit", "bench_boost"])
        self.assertEqual(chips["used_this_half"], ["wildcard", "triple_captain"])

    def test_second_half_brings_a_fresh_set(self):
        # GW19 is finished, so the next deadline is GW20.
        chips = self.chips(19, [("wildcard", 6), ("bboost", 19)])
        self.assertEqual(chips["available"], ["wildcard", "free_hit", "bench_boost", "triple_captain"])
        self.assertEqual(chips["used_this_half"], [])

    def test_planner_requests_fill_chips_from_history(self):
        import main
        with mock.patch.object(main.fpl_service, "get_manager_chips", return_value={"used_this_half": ["wildcard"]}):
            self.assertEqual(main._chips_used(7706676, []), ["wildcard"])
            self.assertEqual(main._chips_used(7706676, ["free_hit"]), ["free_hit"])
        self.assertEqual(main._chips_used(None, []), [])


class PlanningGameweekTests(unittest.TestCase):
    def test_plans_from_the_next_deadline_while_matches_are_still_on(self):
        from main import get_active_gameweek, get_planning_gameweek
        live = [{"id": 7, "is_current": True, "is_next": False, "finished": False},
                {"id": 8, "is_current": False, "is_next": True, "finished": False}]
        self.assertEqual(get_planning_gameweek(live), 8)
        self.assertEqual(get_active_gameweek(live), 7)

    def test_last_gameweek_and_preseason(self):
        from main import get_planning_gameweek
        self.assertEqual(get_planning_gameweek([{"id": 38, "is_current": True, "is_next": False, "finished": False}]), 38)
        self.assertEqual(get_planning_gameweek([{"id": 1, "is_current": False, "is_next": True, "finished": False}]), 1)


class ComparisonSummaryTests(unittest.TestCase):
    def test_infeasible_plan_is_not_scored_as_zero(self):
        class Solution:
            status = "Infeasible"
            total_expected_points = 0

        summary = _comparison_entry(Solution(), 0, 0, 0, True)
        self.assertFalse(summary["feasible"])
        self.assertIsNone(summary["net_xp"])
        self.assertFalse(summary["recommended"])


class PredictionPoolTests(unittest.TestCase):
    def test_the_squad_stays_in_and_the_pool_is_capped(self):
        from advanced_optimizer import prediction_pool
        rows = []
        for pos in (1, 2, 3, 4):
            for i in range(10):
                rows.append({"id": pos * 100 + i, "element_type": pos, "points_per_game": i})
        # The squad player is the lowest points-per-game midfielder.
        squad_id = 300
        pool = prediction_pool(pd.DataFrame(rows), [squad_id], per_position=3)
        self.assertIn(squad_id, set(pool['id']))
        self.assertEqual(len(pool), 3 * 4 + 1)
        self.assertFalse((pool['id'] == 301).any())


class PriorSeasonSwitchTests(unittest.TestCase):
    def test_prior_season_load_can_be_turned_off(self):
        from prior_season import PriorSeasonStats
        stats = PriorSeasonStats()
        started = []
        stats._loader = None

        def explode(*_args, **_kwargs):
            started.append(True)
            raise AssertionError("last season should not be fetched")

        stats._load = explode
        static = {"events": [{"deadline_time": "2026-08-21T17:30:00Z"}], "elements": [{"id": 1, "code": 1}]}
        with mock.patch.dict("os.environ", {"LOAD_PRIOR_SEASON": "0"}):
            stats.attach(static, wait=5)
        self.assertEqual(started, [])
        self.assertNotIn("prior_points", static["elements"][0])

    def test_a_saved_season_is_applied_without_a_fetch(self):
        from prior_season import PriorSeasonStats
        with tempfile.TemporaryDirectory() as cache_dir:
            path = os.path.join(cache_dir, "prior_season_2025-26.json")
            with open(path, "w") as f:
                json.dump({"154561": {"prior_points": 162, "prior_minutes": 3330, "prior_starts": 37}}, f)
            stats = PriorSeasonStats(cache_dir=cache_dir)
            stats._load = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("network fetch"))
            static = {"events": [{"deadline_time": "2026-08-21T17:30:00Z"}],
                      "elements": [{"id": 1, "code": 154561}]}
            stats.attach(static, wait=0)
        self.assertEqual(static["elements"][0]["prior_points"], 162)


class LatestDataTests(unittest.TestCase):
    def test_a_request_does_not_wait_on_last_season(self):
        service = FPLService()
        service._update_cache("bootstrap-static", {"elements": [], "events": []})
        service._update_cache("fixtures", [])

        def slow_attach(static, wait=0):
            time.sleep(wait)

        service.prior_season.attach = slow_attach
        started = time.time()
        data = service.get_latest_data()
        self.assertLess(time.time() - started, 0.5)
        self.assertEqual(data["fixtures"], [])

        started = time.time()
        service.get_latest_data(prior_wait=0.3)
        self.assertGreaterEqual(time.time() - started, 0.3)

    def test_bootstrap_is_cached_before_last_season_is_waited_on(self):
        service = FPLService()
        order = []

        def get_json(url):
            order.append(url)
            if url.endswith("/bootstrap-static/"):
                return {"elements": [], "events": [{"deadline_time": "2026-08-21T17:30:00Z"}]}
            return []

        def attach(static, wait=0):
            order.append("cached" if service._is_cache_valid("bootstrap-static") else "missing")

        service._get_json = get_json
        service.prior_season.attach = attach
        service.get_latest_data()
        self.assertEqual(order[-1], "cached")


class StrategyAndRobustTests(unittest.TestCase):
    def test_differential_only_helps_players_under_ten_percent(self):
        from advanced_optimizer import ownership_adjustment
        self.assertAlmostEqual(ownership_adjustment(0, "differential"), 1.5)
        self.assertEqual(ownership_adjustment(10, "differential"), 0)
        self.assertEqual(ownership_adjustment(40, "differential"), 0)
        self.assertEqual(ownership_adjustment(0, "standard"), 0)

    def test_template_rewards_the_popular_and_trims_the_rare(self):
        from advanced_optimizer import ownership_adjustment
        self.assertEqual(ownership_adjustment(20, "template"), 0)
        self.assertGreater(ownership_adjustment(50, "template"), ownership_adjustment(25, "template"))
        self.assertAlmostEqual(ownership_adjustment(50, "template"), 1.2)
        self.assertEqual(ownership_adjustment(15, "template"), 0)
        self.assertLess(ownership_adjustment(0, "template"), 0)
        self.assertAlmostEqual(ownership_adjustment(0, "template"), -0.6)

    def test_robust_charges_the_riskier_projection_more(self):
        from advanced_optimizer import robust_points
        safe = robust_points(5, 0.75, 0.3)
        risky = robust_points(5, 4.0, 0.3)
        self.assertAlmostEqual(safe, 4.775)
        self.assertAlmostEqual(risky, 3.8)
        self.assertGreater(safe, risky)
        self.assertEqual(robust_points(2, 100, 0.3), 0)


if __name__ == "__main__":
    unittest.main()
