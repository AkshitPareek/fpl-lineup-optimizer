import tempfile
import threading
import unittest

import pandas as pd

from point_predictor import PointPredictor
from prior_season import PriorSeasonStats, previous_season_name, totals_from_summary


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

    def test_thin_or_missing_prior_seasons_shrink_toward_the_newcomer_baseline(self):
        predictor = make_predictor([player()])
        # Six games at 7.0 against six games at the 3.0 baseline.
        self.assertEqual(predictor.blended_points_per_game(pd.Series(player(points_per_game="7.0"))), 5.0)
        thin = pd.Series(player(points_per_game="7.0", prior_points=30, prior_minutes=400, prior_starts=4))
        self.assertEqual(predictor.blended_points_per_game(thin), 5.0)

    def test_two_big_games_do_not_make_a_captain(self):
        lucky = pd.Series(player(points_per_game="10.0", minutes=180, starts=2))
        self.assertEqual(make_predictor([player()]).blended_points_per_game(lucky), 4.75)


class FormIndependenceTests(unittest.TestCase):
    def test_a_cold_month_does_not_cut_the_projection(self):
        predictor = make_predictor([player()])
        hot = predictor.predict_gameweek(pd.Series(player(form="10.0")), 7)[0]
        cold = predictor.predict_gameweek(pd.Series(player(form="1.0")), 7)[0]
        self.assertAlmostEqual(hot, cold)


class MinutesTests(unittest.TestCase):
    def test_live_data_without_recent_minutes_uses_season_minutes(self):
        predictor = make_predictor([player()])
        regular = predictor.predict_gameweek(pd.Series(player()), 7)[0]
        fringe = predictor.predict_gameweek(pd.Series(player(minutes=27, starts=0)), 7)[0]
        self.assertAlmostEqual(predictor.average_minutes_per_gameweek(pd.Series(player(minutes=27))), 4.5)
        self.assertLess(fringe, 0.35 * regular)

    def test_tiny_understat_samples_are_ignored(self):
        predictor = make_predictor([player()])
        predictor._understat_enriched = True
        cameo = pd.Series(player(minutes=27, understat_matched=True, understat_minutes=27,
                                 understat_xG_per_90=12.0, understat_xA_per_90=3.0))
        self.assertEqual(predictor._get_xg_per_90(cameo), PointPredictor.BASELINE_XG[3])
        self.assertEqual(predictor._get_xa_per_90(cameo), PointPredictor.BASELINE_XA[3])


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

    def test_a_named_return_date_is_zero_until_that_week(self):
        fixtures = [
            {"event": 9, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3,
             "kickoff_time": "2026-10-31T14:00:00Z"},
            {"event": 10, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3,
             "kickoff_time": "2026-11-07T15:00:00Z"},
            {"event": 11, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3,
             "kickoff_time": "2026-11-21T15:00:00Z"},
        ]
        predictor = make_predictor([player()], fixtures=fixtures, current_gw=9)
        hurt = pd.Series(player(status="i", chance_of_playing_next_round=0,
                                news="Hamstring injury - Expected back 1 Nov"))
        self.assertEqual(predictor.availability(hurt, 9), 0.0)
        self.assertAlmostEqual(predictor.availability(hurt, 10), 0.75)
        self.assertGreater(predictor.availability(hurt, 11), 0.75)
        self.assertEqual(predictor.predict_gameweek(hurt, 9)[0], 0.0)
        self.assertGreater(predictor.predict_gameweek(hurt, 10)[0], 0.0)

    def test_unknown_return_misses_three_weeks_then_comes_back(self):
        predictor = make_predictor([player()], current_gw=9)
        hurt = pd.Series(player(status="i", chance_of_playing_next_round=0,
                                news="Back injury - Unknown return date"))
        self.assertEqual([predictor.availability(hurt, gw) for gw in (9, 10, 11)], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(predictor.availability(hurt, 12), 0.5)

    def test_a_player_who_has_left_stays_out(self):
        predictor = make_predictor([player()], current_gw=9)
        gone = pd.Series(player(status="u", chance_of_playing_next_round=0,
                                news="Has joined Al Hilal permanently"))
        self.assertEqual(predictor.availability(gone, 16), 0.0)


class FixtureCountTests(unittest.TestCase):
    def test_blank_gameweek_scores_nothing(self):
        fixtures = [{"event": 7, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3}]
        predictor = make_predictor([player()], fixtures=fixtures)
        self.assertEqual(predictor.predict_gameweek(pd.Series(player()), 8)[0], 0.0)

    def test_double_gameweek_counts_both_matches_in_full(self):
        home = {"event": 8, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3}
        away = {"event": 8, "team_h": 2, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3}
        mid = pd.Series(player())
        single = make_predictor([player()], fixtures=[home]).predict_gameweek(mid, 8)
        double = make_predictor([player()], fixtures=[home, away]).predict_gameweek(mid, 8)
        away_only = make_predictor([player()], fixtures=[away]).predict_gameweek(mid, 8)
        # Home and away are both counted. They are not the same fixture, so the
        # total is the sum of the two matches, and appearance still doubles.
        self.assertAlmostEqual(double[0], single[0] + away_only[0])
        for part in ("appearance", "bonus", "cbit_bonus"):
            self.assertAlmostEqual(double[1][part], 2 * single[1][part])

    def test_double_gameweek_caps_are_per_match(self):
        fixtures = [{"event": 8, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3},
                    {"event": 8, "team_h": 2, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3}]
        striker = pd.Series(player(element_type=4, expected_goals="12.0"))
        predictor = make_predictor([player()], fixtures=fixtures)
        self.assertGreater(predictor.calculate_xg_points(striker, 8), 9.0)


def team_row(team_id, attack_home, attack_away, defence_home, defence_away):
    return {
        "id": team_id, "name": f"Team {team_id}",
        "strength_attack_home": attack_home, "strength_attack_away": attack_away,
        "strength_defence_home": defence_home, "strength_defence_away": defence_away,
    }


class FixtureSensitivityTests(unittest.TestCase):
    def test_later_weeks_keep_this_weeks_calibration(self):
        fixtures = [
            {"event": gw, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3}
            for gw in (8, 13)
        ]
        predictor = make_predictor([player()], fixtures=fixtures, current_gw=8)
        now = predictor.predict_gameweek(pd.Series(player()), 8)
        later = predictor.predict_gameweek(pd.Series(player()), 13)
        self.assertEqual(now[1]['calibration'], later[1]['calibration'])
        self.assertAlmostEqual(now[0], later[0])

    def test_a_hard_away_trip_cuts_a_defender(self):
        teams = pd.DataFrame([
            team_row(1, 1100, 1050, 1100, 1000),
            team_row(2, 1450, 1350, 1400, 1300),
            team_row(3, 900, 850, 950, 900),
        ])
        defender = pd.Series(player(element_type=2, web_name="Def", points_per_game="4.5",
                                    expected_goals="0.4", expected_assists="0.3", goals_scored=0, assists=0))
        hard = [{"event": 9, "team_h": 2, "team_a": 1, "team_h_difficulty": 5, "team_a_difficulty": 5}]
        easy = [{"event": 9, "team_h": 1, "team_a": 3, "team_h_difficulty": 2, "team_a_difficulty": 2}]
        hard_xp = PointPredictor(pd.DataFrame([player()]), teams, hard, 9, use_understat=False).predict_gameweek(defender, 9)[0]
        easy_xp = PointPredictor(pd.DataFrame([player()]), teams, easy, 9, use_understat=False).predict_gameweek(defender, 9)[0]
        self.assertGreater(easy_xp, hard_xp * 1.12)
        self.assertGreater(easy_xp - hard_xp, 0.6)

    def test_missing_team_strengths_fall_back_to_fixture_difficulty(self):
        teams = pd.DataFrame([
            team_row(1, 0, 0, 0, 0),
            team_row(2, 0, 0, 0, 0),
        ])
        defender = pd.Series(player(element_type=2, web_name="Def"))
        hard = [{"event": 9, "team_h": 2, "team_a": 1, "team_h_difficulty": 2, "team_a_difficulty": 5}]
        easy = [{"event": 9, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4}]
        hard_xp, hard_br = PointPredictor(pd.DataFrame([player()]), teams, hard, 9, use_understat=False).predict_gameweek(defender, 9)
        easy_xp, easy_br = PointPredictor(pd.DataFrame([player()]), teams, easy, 9, use_understat=False).predict_gameweek(defender, 9)
        self.assertAlmostEqual(hard_br['fixture_factor'], 0.70)
        self.assertAlmostEqual(easy_br['fixture_factor'], 1.12)
        self.assertGreater(easy_xp, hard_xp)

    def test_an_attackers_fixture_moves_the_projection(self):
        teams = pd.DataFrame([
            team_row(1, 1200, 1150, 1100, 1050),
            team_row(2, 1100, 1100, 1450, 1400),
            team_row(3, 1000, 950, 850, 800),
        ])
        mid = pd.Series(player())
        fortress = [{"event": 9, "team_h": 2, "team_a": 1, "team_h_difficulty": 5, "team_a_difficulty": 4}]
        leaky = [{"event": 9, "team_h": 1, "team_a": 3, "team_h_difficulty": 2, "team_a_difficulty": 2}]
        hard_xp = PointPredictor(pd.DataFrame([player()]), teams, fortress, 9, use_understat=False).predict_gameweek(mid, 9)[0]
        easy_xp = PointPredictor(pd.DataFrame([player()]), teams, leaky, 9, use_understat=False).predict_gameweek(mid, 9)[0]
        self.assertGreater(easy_xp, hard_xp * 1.08)


class UnderstatSeasonTests(unittest.TestCase):
    def test_season_comes_from_the_fixture_list(self):
        fixtures = [{"event": 7, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3,
                     "kickoff_time": "2026-10-17T11:30:00Z"},
                    {"event": 1, "team_h": 2, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3,
                     "kickoff_time": "2026-08-21T19:00:00Z"}]
        self.assertEqual(make_predictor([player()], fixtures=fixtures)._season_start_year(), "2026")


class PriorSeasonStatsTests(unittest.TestCase):
    def test_wait_joins_a_load_that_is_already_running(self):
        release = threading.Event()

        class SlowStats(PriorSeasonStats):
            def _fetch_one(self, element, season_name):
                release.wait(5)
                return str(element["code"]), {"prior_points": 162, "prior_minutes": 3330, "prior_starts": 37}

        with tempfile.TemporaryDirectory() as cache_dir:
            stats = SlowStats(cache_dir=cache_dir)
            static = {"events": [{"deadline_time": "2026-08-21T17:30:00Z"}],
                      "elements": [{"id": 1, "code": 154561}]}
            stats.attach(static)
            self.assertNotIn("prior_points", static["elements"][0])
            threading.Timer(0.05, release.set).start()
            stats.attach(static, wait=5)
            self.assertEqual(static["elements"][0]["prior_points"], 162)


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
