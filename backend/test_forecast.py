import unittest

from price_forecast import (
    forecast_price_path, guaranteed_sale, infer_purchase_price, sale_proceeds, transfer_elapsed,
)
from squad_value import autosub_points, best_lineup, blank_probability, slot_weight


class PriceForecastTests(unittest.TestCase):
    def test_purchase_price_matches_half_the_rise(self):
        # Bought at 7.5, now 7.8, sells for 7.6.
        self.assertEqual(infer_purchase_price(76, 78), 75)
        self.assertEqual(infer_purchase_price(60, 60, season_start_tenths=65), 65)

    def test_projected_transfers_move_the_price_then_fade(self):
        # 20% owned of 10m managers is 2m owners. A rise step is 7.5% of that, 150,000.
        # 450,000 net transfers already in, with the window half gone, projects 900,000: six
        # steps, capped at three this week. Half of that projected net remains next week.
        player = {
            "now_cost": 100, "selected_by_percent": 20,
            "transfers_in_event": 500000, "transfers_out_event": 50000, "cost_change_event": 0,
        }
        path = forecast_price_path(player, weeks=4, total_managers=10_000_000, elapsed=0.5)
        self.assertEqual(path, [103, 106, 107, 107])

    def test_a_later_sale_keeps_half_the_rise(self):
        # Bought at 10.0 and sold at 10.6 keeps 10.3. Bought at 10.3 keeps 10.4.
        # The plan can count on the lower of those.
        self.assertEqual(sale_proceeds(100, 106), 103)
        self.assertEqual(sale_proceeds(106, 100), 100)
        self.assertEqual(guaranteed_sale([100, 103, 106], 2), 103)

    def test_no_transfer_market_leaves_the_price_alone(self):
        self.assertEqual(forecast_price_path({"now_cost": 55}, 3, total_managers=0), [55, 55, 55])

    def test_elapsed_window_is_a_fraction(self):
        events = [
            {"id": 7, "deadline_time": "2026-10-01T17:30:00Z"},
            {"id": 8, "is_next": True, "deadline_time": "2026-10-18T10:00:00Z"},
        ]
        from datetime import datetime, timezone
        midway = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
        self.assertGreater(transfer_elapsed(events, midway), 0.4)
        self.assertLess(transfer_elapsed(events, midway), 0.6)


class AutosubTests(unittest.TestCase):
    def test_a_full_game_player_does_not_blank_and_a_flagged_one_does(self):
        playing = {"minutes": 360, "chance_of_playing_next_round": None}
        self.assertEqual(blank_probability(playing, gameweek=5, current_gw=5), 0.0)
        ruled_out = {"minutes": 360, "chance_of_playing_next_round": 0}
        self.assertEqual(blank_probability(ruled_out, 5, 5), 1.0)
        unknown = {}
        self.assertEqual(blank_probability(unknown, 5, 5), 0.0)

    def test_substitutes_are_used_in_order_with_the_blank_count(self):
        # One outfield starter is certain to blank; two bench players score 5 and 3.
        xp = {1: 0, 2: 4, 11: 6, 12: 5, 13: 3}
        blank = {1: 1.0, 2: 1.0, 11: 0.0, 12: 0.0, 13: 0.0}
        position = {1: 1, 2: 2, 11: 1, 12: 3, 13: 2}.get
        added = autosub_points([2], [12, 13], xp.get, blank.get, position)
        # One certain outfield blank brings on only the first substitute.
        self.assertAlmostEqual(added, 5.0)
        added = autosub_points([1, 2], [11, 12], xp.get, blank.get, position)
        self.assertAlmostEqual(added, 6.0 + 5.0)

    def test_the_shown_lineup_keeps_the_blankers_so_the_substitutes_play(self):
        # The two defenders on 2 points will not play. Leaving them in the eleven brings on
        # the two 9-point midfielders. Starting those midfielders instead drops the blanks,
        # and the bench then scores nothing.
        players = list(range(1, 16))
        xp = {1: 4, 2: 0, 3: 6, 4: 6, 5: 6, 6: 2, 7: 2, 8: 9, 9: 9, 10: 1, 11: 1, 12: 1, 13: 6, 14: 1, 15: 1}
        blank = {i: 0.0 for i in players}
        blank[6] = 1.0
        blank[7] = 1.0
        position = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 4, 14: 4, 15: 4}
        starters, captain = best_lineup(
            players, xp.get, blank.get, position.get, {1: 1, 2: 3, 3: 2, 4: 1}, {1: 1, 2: 5, 3: 5, 4: 3})
        # The blankers stay in, so the bench still comes on. The 9-point players start as well:
        # a starter's points are counted in full, and the blanks bring on whoever is left.
        self.assertTrue({6, 7, 8, 9}.issubset(starters))
        self.assertIn(captain, starters)
        self.assertEqual(xp[captain], 9)

    def test_later_substitutes_only_take_the_overflow(self):
        self.assertEqual(slot_weight(0.25, 1), 0.25)
        self.assertEqual(slot_weight(0.25, 2), 0.0)
        self.assertAlmostEqual(slot_weight(1.4, 2), 0.4)


if __name__ == "__main__":
    unittest.main()
