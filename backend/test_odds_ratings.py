import unittest
from math import exp, factorial

import pandas as pd

import odds_ratings
from odds_ratings import TeamRatings, goal_expectations, team_ids
from point_predictor import PointPredictor


def fair_odds(home_rate, away_rate):
    """Margin-free decimal odds for home/draw/away and over/under 2.5 under independent Poisson goals."""
    pmf = lambda rate: [exp(-rate) * rate ** k / factorial(k) for k in range(11)]
    h, a = pmf(home_rate), pmf(away_rate)
    home = sum(h[i] * a[j] for i in range(11) for j in range(11) if i > j)
    away = sum(h[i] * a[j] for i in range(11) for j in range(11) if i < j)
    over = sum(h[i] * a[j] for i in range(11) for j in range(11) if i + j > 2)
    return 1 / home, 1 / (1 - home - away), 1 / away, 1 / over, 1 / (1 - over)


TEAMS = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7],
                      "name": ["Strong", "Middle", "Weak", "Hull City", "Upper", "Lower", "Bottom"]})
LEVEL = {"Strong": 0.5, "Upper": 0.25, "Middle": 0.0, "Lower": -0.15, "Weak": -0.3, "Bottom": -0.4}


def season():
    """A double round robin priced from each club's level, a match every three days."""
    rows, day = [], pd.Timestamp("2025-08-16")
    for home in LEVEL:
        for away in LEVEL:
            if home != away:
                gap = LEVEL[home] - LEVEL[away]
                rows.append({"date": day, "home": home, "away": away,
                             "home_xg": 1.5 * exp(gap), "away_xg": 1.2 * exp(-gap)})
                day += pd.Timedelta(days=3)
    return pd.DataFrame(rows)


LEAGUE = season()
AS_OF = LEAGUE["date"].max() + pd.Timedelta(days=1)


class GoalExpectationTests(unittest.TestCase):
    def test_recovers_the_rates_behind_fair_odds(self):
        for rates in ((1.8, 1.1), (0.9, 2.4), (1.35, 1.35)):
            home, away = goal_expectations(*fair_odds(*rates))
            self.assertAlmostEqual(home, rates[0], delta=0.03)
            self.assertAlmostEqual(away, rates[1], delta=0.03)

    def test_bookmaker_margin_is_removed(self):
        odds = [o / 1.06 for o in fair_odds(1.6, 1.2)]
        home, away = goal_expectations(*odds)
        self.assertAlmostEqual(home, 1.6, delta=0.05)
        self.assertAlmostEqual(away, 1.2, delta=0.05)

    def test_unpriced_match_is_skipped(self):
        self.assertIsNone(goal_expectations(None, 3.4, 2.1))


class TeamRatingTests(unittest.TestCase):
    def setUp(self):
        self.ratings = TeamRatings(LEAGUE, TEAMS, AS_OF)

    def test_football_data_names_map_to_fpl_clubs(self):
        self.assertEqual(team_ids(TEAMS, ["Hull", "Strong", "Nowhere"]), {"Hull": 4, "Strong": 1})
        fpl = pd.DataFrame({"id": [13, 18], "name": ["Man Utd", "Spurs"]})
        self.assertEqual(team_ids(fpl, ["Man United", "Tottenham"]), {"Man United": 13, "Tottenham": 18})

    def test_stronger_club_scores_more_and_concedes_less(self):
        strong, weak = self.ratings.typical_goals(1), self.ratings.typical_goals(3)
        self.assertGreater(strong[0], weak[0])
        self.assertLess(strong[1], weak[1])
        self.assertGreater(self.ratings.expected_goals(1, 3, True)[0], self.ratings.expected_goals(1, 3, False)[0])

    def test_club_without_matches_starts_with_the_weakest(self):
        unseen = self.ratings.typical_goals(4)
        self.assertLess(unseen[0], self.ratings.typical_goals(2)[0])
        self.assertGreater(unseen[1], self.ratings.typical_goals(2)[1])

    def test_published_odds_override_the_ratings(self):
        upcoming = pd.DataFrame([{"date": AS_OF, "home": "Weak", "away": "Strong",
                                  "home_xg": 1.9, "away_xg": 0.7}])
        ratings = TeamRatings(LEAGUE, TEAMS, AS_OF, upcoming)
        self.assertEqual(ratings.expected_goals(3, 1, True), (1.9, 0.7))
        self.assertEqual(ratings.expected_goals(1, 3, False), (0.7, 1.9))

    def test_later_matches_are_ignored(self):
        early = TeamRatings(LEAGUE, TEAMS, pd.Timestamp("2025-08-20"))
        self.assertEqual(early.matches_used, 2)


class PredictorOddsTests(unittest.TestCase):
    def predictor(self, ratings):
        players = pd.DataFrame([{"id": 1, "team": 1, "element_type": 2}])
        fixtures = [{"event": 9, "team_h": 1, "team_a": 3, "team_h_difficulty": 2, "team_a_difficulty": 4},
                    {"event": 10, "team_h": 2, "team_a": 1, "team_h_difficulty": 4, "team_a_difficulty": 3}]
        teams = TEAMS.assign(strength_attack_home=0, strength_attack_away=0,
                             strength_defence_home=0, strength_defence_away=0)
        return PointPredictor(players, teams, fixtures, 9, use_understat=False, team_ratings=ratings)

    def test_fixture_is_measured_against_the_club_itself(self):
        predictor = self.predictor(TeamRatings(LEAGUE, TEAMS, AS_OF))
        easy = predictor.attack_fixture_multiplier(3, True, team_id=1)
        hard = predictor.attack_fixture_multiplier(1, True, team_id=3)
        # Both are home games against a club of the other's level, so neither
        # side's own strength should push its multiplier far from the other's.
        self.assertGreater(easy, 1.0)
        self.assertLess(hard, 1.0)
        self.assertLess(abs((easy - 1) + (hard - 1)), 0.3)

    def test_clean_sheet_chance_comes_from_goals_conceded(self):
        ratings = TeamRatings(LEAGUE, TEAMS, AS_OF)
        predictor = self.predictor(ratings)
        player = predictor.players.iloc[0]
        conceded = ratings.expected_goals(1, 3, True)[1]
        self.assertAlmostEqual(predictor.calculate_clean_sheet_prob(player, 9),
                               exp(-conceded) * PointPredictor.POINTS_CLEAN_SHEET[2], places=6)

    def test_difficulty_is_used_without_ratings(self):
        predictor = self.predictor(None)
        self.assertEqual(predictor.attack_fixture_multiplier(3, True, 2, team_id=1), PointPredictor.FDR_ATTACK[2])


class LiveRatingTests(unittest.TestCase):
    def test_no_fixtures_means_no_ratings(self):
        self.assertIsNone(odds_ratings.live_ratings(TEAMS, []))


if __name__ == "__main__":
    unittest.main()
