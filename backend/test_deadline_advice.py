import unittest

import pandas as pd

from main import _deadline_advice
from ownership_tracker import OwnershipTracker
from price_forecast import price_alerts, price_change_chance

TOTAL = 10_000_000
TEAMS = pd.DataFrame([{"id": 1, "name": "Team 1", "short_name": "ONE"}])


def element(pid, owned_pct=10.0, net=0, moved=0, **overrides):
    base = {
        "id": pid, "web_name": f"P{pid}", "team": 1, "element_type": 3, "now_cost": 80, "form": "5.0",
        "selected_by_percent": str(owned_pct), "transfers_in_event": max(net, 0),
        "transfers_out_event": max(-net, 0), "cost_change_event": moved,
    }
    base.update(overrides)
    return base


class PriceChanceTests(unittest.TestCase):
    def test_heavy_net_transfers_in_make_a_rise_likely(self):
        rise, fall = price_change_chance(400_000, 1_000_000)
        self.assertGreater(rise, 0.7)
        self.assertLess(fall, 0.05)

    def test_a_quiet_week_predicts_no_change(self):
        rise, fall = price_change_chance(2_000, 500_000)
        self.assertLess(rise, 0.05)
        self.assertLess(fall, 0.1)

    def test_a_player_sold_by_a_large_share_of_owners_is_likely_to_fall(self):
        _, fall = price_change_chance(-300_000, 600_000)
        self.assertGreater(fall, 0.5)


class PriceAlertTests(unittest.TestCase):
    def test_a_planned_buy_likely_to_rise_and_a_squad_player_likely_to_fall_are_flagged(self):
        players = pd.DataFrame([
            element(1, owned_pct=8.0, net=-300_000),
            element(2, owned_pct=20.0, net=600_000),
            element(3, owned_pct=20.0, net=1_000),
        ])
        alerts = price_alerts(players, TOTAL, squad_ids=[1, 3], buy_ids=[2])
        self.assertEqual({(a["id"], a["direction"]) for a in alerts}, {(1, "fall"), (2, "rise")})

    def test_a_player_who_already_rose_this_gameweek_is_not_flagged_again(self):
        players = pd.DataFrame([element(2, owned_pct=20.0, net=600_000, moved=1)])
        self.assertEqual(price_alerts(players, TOTAL, squad_ids=[], buy_ids=[2]), [])

    def test_no_manager_count_means_no_alerts(self):
        players = pd.DataFrame([element(2, owned_pct=20.0, net=600_000)])
        self.assertEqual(price_alerts(players, None, squad_ids=[], buy_ids=[2]), [])


class CaptainOptionTests(unittest.TestCase):
    def xi(self):
        return [{"id": 1, "expected_points": 7.0}, {"id": 2, "expected_points": 6.6},
                {"id": 3, "expected_points": 3.0}]

    def test_options_keep_the_points_order(self):
        players = pd.DataFrame([element(1, owned_pct=60.0, now_cost=140, element_type=4),
                                element(2, owned_pct=4.0), element(3, owned_pct=1.0)])
        options = OwnershipTracker(players, TEAMS).captain_options(self.xi(), top_n=2)
        self.assertEqual([o["id"] for o in options], [1, 2])
        self.assertAlmostEqual(options[1]["behind_best"], 0.4)

    def test_a_template_captain_moves_you_less_than_a_differential(self):
        players = pd.DataFrame([element(1, owned_pct=60.0, now_cost=140, element_type=4),
                                element(2, owned_pct=4.0), element(3, owned_pct=1.0)])
        template, differential = OwnershipTracker(players, TEAMS).captain_options(self.xi(), top_n=2)
        self.assertLess(template["swing"], 1.2)
        self.assertGreater(differential["swing"], 1.9)


class DeadlineAdviceTests(unittest.TestCase):
    def test_advice_is_added_without_touching_the_plan(self):
        players = pd.DataFrame([element(1, owned_pct=60.0, net=-2_000_000), element(2, owned_pct=4.0, net=500_000),
                                element(3)])
        plan = {"gameweek": 10, "captain_id": 1, "chip_used": None,
                "starting_xi": [{"id": 1, "expected_points": 7.0}, {"id": 2, "expected_points": 6.0}],
                "bench": [{"id": 3, "expected_points": 1.0}],
                "transfers": {"in": [{"id": 2}], "out": [{"id": 4}]}}
        response = {"gameweek_plans": [plan]}
        _deadline_advice(response, players, TEAMS, TOTAL)
        self.assertEqual(plan["captain_id"], 1)
        self.assertEqual([o["id"] for o in response["captain_options"]], [1, 2])
        self.assertEqual({(a["id"], a["direction"]) for a in response["price_alerts"]}, {(1, "fall"), (2, "rise")})


if __name__ == "__main__":
    unittest.main()
