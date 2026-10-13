import unittest

import pandas as pd

from point_predictor import PointPredictor
from prior_season import previous_season_name, totals_from_summary


def make_predictor(players, fixtures=None, current_gw=7):
    teams = pd.DataFrame([
        {"id": t, "name": f"Team {t}", "strength_attack_home": 1100, "strength_attack_away": 1100,
         "strength_defence_home": 1100, "strength_defence_away": 1100}
        for t in (1, 2)
    ])
    if fixtures is None:
        fixtures = [{"event": gw, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3}
                    for gw in range(current_gw, current_gw + 8)]
    return PointPredictor(pd.DataFrame(players), teams, fixtures, current_gw, use_understat=False)


def player(**overrides):
    base = {
        "id": 1, "web_name": "Mid", "team": 1, "element_type": 3, "now_cost": 70,
        "minutes": 540, "starts": 6, "total_points": 30, "points_per_game": "5.0", "form": "5.0",
        "goals_scored": 2, "assists": 1, "expected_goals": "2.0", "expected_assists": "1.0",
        "influence": "150.0", "ict_index": "40.0", "chance_of_playing_next_round": None,
    }
    base.update(overrides)
    return base


class BlendedPointsPerGameTests(unittest.TestCase):
    def test_shrinks_this_season_toward_last_season(self):
        p = pd.Series(player(points_per_game="9.0", prior_points=152, prior_minutes=3420, prior_starts=38))
        # Six games at 9.0 and six "prior games" at 4.0 meet halfway.
        self.assertAlmostEqual(make_predictor([player()]).blended_points_per_game(p), 6.5)

    def test_ignores_thin_or_missing_prior_seasons(self):
        predictor = make_predictor([player()])
        self.assertEqual(predictor.blended_points_per_game(pd.Series(player(points_per_game="7.0"))), 7.0)
        thin = pd.Series(player(points_per_game="7.0", prior_points=30, prior_minutes=400, prior_starts=4))
        self.assertEqual(predictor.blended_points_per_game(thin), 7.0)


class FormIndependenceTests(unittest.TestCase):
    def test_a_cold_month_does_not_cut_the_projection(self):
        predictor = make_predictor([player()])
        hot = predictor.predict_gameweek(pd.Series(player(form="10.0")), 7)[0]
        cold = predictor.predict_gameweek(pd.Series(player(form="1.0")), 7)[0]
        self.assertAlmostEqual(hot, cold)


class AvailabilityTests(unittest.TestCase):
    def test_doubts_recover_over_the_horizon(self):
        predictor = make_predictor([player()])
        doubtful = pd.Series(player(chance_of_playing_next_round=75))
        self.assertEqual(
            [predictor.availability(doubtful, gw) for gw in (7, 8, 9)],
            [0.75, 0.875, 0.9375],
        )

    def test_ruled_out_players_stay_out(self):
        predictor = make_predictor([player()])
        out = pd.Series(player(chance_of_playing_next_round=0))
        self.assertEqual(predictor.predict_gameweek(out, 9)[0], 0.0)


class FixtureCountTests(unittest.TestCase):
    def test_blank_gameweek_scores_nothing(self):
        fixtures = [{"event": 7, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3}]
        predictor = make_predictor([player()], fixtures=fixtures)
        self.assertEqual(predictor.predict_gameweek(pd.Series(player()), 8)[0], 0.0)

    def test_double_gameweek_scores_more(self):
        fixtures = [{"event": 7, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3},
                    {"event": 8, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3},
                    {"event": 8, "team_h": 2, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3}]
        predictor = make_predictor([player()], fixtures=fixtures)
        single = predictor.predict_gameweek(pd.Series(player()), 7)[0]
        double = predictor.predict_gameweek(pd.Series(player()), 8)[0]
        self.assertGreater(double, 1.5 * single)


class PriorSeasonHelperTests(unittest.TestCase):
    def test_previous_season_name_comes_from_the_first_deadline(self):
        events = [{"deadline_time": "2026-08-21T17:30:00Z"}, {"deadline_time": "2027-05-23T13:30:00Z"}]
        self.assertEqual(previous_season_name(events), "2025/26")

    def test_totals_use_only_the_named_season(self):
        summary = {"history_past": [
            {"season_name": "2024/25", "total_points": 200, "minutes": 3000, "starts": 34},
            {"season_name": "2025/26", "total_points": 162, "minutes": 3330, "starts": 37},
        ]}
        self.assertEqual(totals_from_summary(summary, "2025/26"),
                         {"prior_points": 162, "prior_minutes": 3330, "prior_starts": 37})
        self.assertIsNone(totals_from_summary({"history_past": summary["history_past"][:1]}, "2025/26"))


if __name__ == "__main__":
    unittest.main()
