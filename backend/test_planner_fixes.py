import unittest
from unittest import mock

from fpl_service import FPLService, banked_transfers_for_next_deadline, manager_squad_state, selling_price
from main import _comparison_entry


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


if __name__ == "__main__":
    unittest.main()
