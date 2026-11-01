import contextlib
import io
import unittest
from unittest import mock

import pulp

from advanced_optimizer import Chip, MultiPeriodFPLOptimizer, best_xi_points, expected_max_normal
from season_sim import autosub_score
from test_long_horizon import cheapest_top_squad, synthetic_league


def setUpModule():
    # Made-up players never match Understat, so do not ask it.
    patcher = mock.patch('point_predictor.UNDERSTAT_AVAILABLE', False)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def plan(current_gw, gameweeks=5, chips_used=(), adjust=None, bank=5.0):
    """Plan from a squad of cheap players, with money to upgrade, and chips placed by the planner."""
    players, teams, predictions = synthetic_league(current_gw)
    squad = cheapest_top_squad(players)
    if adjust:
        adjust(predictions, squad)
    optimizer = MultiPeriodFPLOptimizer(players, teams, [], current_gw, predictions_df=predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        solution = optimizer.optimize_multi_period(
            budget=4.5 * len(squad) + bank, gameweeks=gameweeks, current_squad_ids=squad,
            chips_used=list(chips_used), max_hits=0, plan_chips=True,
        )
    return squad, solution


def chips(solution):
    return {gp.gameweek: gp.chip_used for gp in solution.gameweek_plans if gp.chip_used}


def best_in_squad(predictions, squad, gw):
    rows = predictions[predictions.id.isin(squad)]
    return int(rows.loc[rows[f'xp_gw{gw}'].idxmax(), 'id'])


class HoldValueTests(unittest.TestCase):
    def test_expected_best_of_normal_draws(self):
        self.assertEqual(expected_max_normal(1), 0.0)
        self.assertAlmostEqual(expected_max_normal(2), 0.5642, places=3)
        self.assertAlmostEqual(expected_max_normal(10), 1.5388, places=3)

    def test_best_xi_meets_the_formation_before_taking_the_best(self):
        squad = ([{'element_type': 1, 'xp': x} for x in (5, 1)]
                 + [{'element_type': 2, 'xp': x} for x in (1, 1, 1, 1, 1)]
                 + [{'element_type': 3, 'xp': x} for x in (9, 9, 9, 9, 9)]
                 + [{'element_type': 4, 'xp': x} for x in (1, 1, 1)])
        xi, bench = best_xi_points(squad)
        # GK 5, three defenders, five midfielders, a forward, one more 1, and the captain's 9 again
        self.assertEqual(xi, 5 + 3 + 45 + 1 + 1 + 9)
        self.assertEqual(bench, 1 + 1 + 1 + 1)


class ChipPlanningTests(unittest.TestCase):
    def test_triple_captain_goes_on_a_standout_week(self):
        def spike(predictions, squad):
            star = best_in_squad(predictions, squad, 7)
            predictions.loc[predictions.id == star, 'xp_gw7'] = 15.0

        # No money to upgrade, so no other week ties with keeping a chip.
        _, solution = plan(5, adjust=spike, bank=0.0)
        self.assertEqual(chips(solution).get(7), Chip.TRIPLE_CAPTAIN)
        self.assertEqual(solution.chip_suggestion['chip'], 'triple_captain')
        self.assertEqual(solution.chip_suggestion['recommended_gameweek'], 7)

    def test_flat_weeks_keep_the_chips_while_the_half_has_time(self):
        # No money to upgrade, so every week of the half looks the same.
        _, solution = plan(5, bank=0.0)
        self.assertEqual(chips(solution), {})
        self.assertEqual({h['chip'] for h in solution.chip_hold},
                         {'triple_captain', 'bench_boost', 'free_hit', 'wildcard'})

    def test_chips_are_played_before_they_expire(self):
        # GW15-19, the last weeks of the half. After a Wildcard the planned squad is
        # already each week's best and a Free Hit gains nothing, so spend it first.
        _, solution = plan(15, chips_used=['wildcard'])
        played = sorted(c.value for c in chips(solution).values())
        self.assertEqual(played, ['bench_boost', 'freehit', 'triple_captain'])
        self.assertEqual(len(chips(solution)), 3)  # one chip a week

    def test_chips_used_this_half_are_not_planned(self):
        _, solution = plan(15, chips_used=['triple_captain', 'bboost', 'wildcard'])
        self.assertEqual([c for c in chips(solution).values()], [Chip.FREE_HIT])

    def test_wildcard_rebuilds_the_squad_before_it_expires(self):
        _, solution = plan(15)
        weeks = {gp.gameweek: gp for gp in solution.gameweek_plans}
        wildcard = next(gw for gw, chip in chips(solution).items() if chip == Chip.WILDCARD)
        self.assertGreater(len(weeks[wildcard].transfers.transfers_in), 1)
        self.assertEqual(weeks[wildcard].transfers.hits_taken, 0)

    def test_free_hit_covers_a_blank_and_the_squad_comes_back(self):
        def blank(predictions, squad):
            predictions.loc[predictions.id.isin(squad), 'xp_gw7'] = 0.0

        # With the Wildcard to hand the plan rebuilds for good instead.
        squad, solution = plan(5, adjust=blank, chips_used=['wildcard'])
        self.assertEqual(chips(solution).get(7), Chip.FREE_HIT)
        weeks = {gp.gameweek: gp for gp in solution.gameweek_plans}
        free_hit = {p['id'] for p in weeks[7].starting_xi + weeks[7].bench}
        before = {p['id'] for p in weeks[6].starting_xi + weeks[6].bench}
        after = {p['id'] for p in weeks[8].starting_xi + weeks[8].bench}
        moves = weeks[8].transfers
        self.assertFalse(free_hit & set(squad))
        self.assertEqual(after, before - {t['id'] for t in moves.transfers_out} | {t['id'] for t in moves.transfers_in})
        self.assertEqual(weeks[7].transfers.hits_taken, 0)

    def test_chip_weeks_count_their_extra_points(self):
        _, solution = plan(15)
        for gp in solution.gameweek_plans:
            xi = sum(p['expected_points'] for p in gp.starting_xi)
            captain = next(p['expected_points'] for p in gp.starting_xi if p['id'] == gp.captain_id)
            bench = sum(p['expected_points'] for p in gp.bench)
            expected = xi + captain
            if gp.chip_used == Chip.TRIPLE_CAPTAIN:
                expected += captain
            if gp.chip_used == Chip.BENCH_BOOST:
                expected += bench
            self.assertAlmostEqual(gp.expected_points, expected, places=6)


class SolverTimeoutTests(unittest.TestCase):
    def test_a_plan_without_chips_when_chip_weeks_run_out_of_time(self):
        solver = MultiPeriodFPLOptimizer._multi_period_solver()

        class OutOfTimeWithChips(type(solver)):
            def actualSolve(self, lp, **kwargs):
                if any(v.name.startswith('chip_') for v in lp.variables()):
                    lp.assignStatus(pulp.LpStatusNotSolved)
                    return lp.status
                return super().actualSolve(lp, **kwargs)

        out_of_time = OutOfTimeWithChips(msg=False)
        with mock.patch.object(MultiPeriodFPLOptimizer, '_multi_period_solver', return_value=out_of_time):
            _, solution = plan(15)
        self.assertEqual(solution.status, 'Optimal')
        self.assertEqual(len(solution.gameweek_plans), 5)
        self.assertEqual(chips(solution), {})


class ChipScoringTests(unittest.TestCase):
    XI = [{'id': i, 'element_type': t} for i, t in enumerate([1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4], start=1)]
    BENCH = [{'id': i, 'element_type': t} for i, t in zip(range(12, 16), [1, 2, 3, 4])]
    PTS = {i: 2 for i in range(1, 16)}
    MINS = {i: 90 for i in range(1, 16)}

    def score(self, chip=None, mins=None):
        return autosub_score(self.XI, self.BENCH, 9, 5, self.PTS, mins or self.MINS, chip)

    def test_triple_captain_counts_the_captain_three_times(self):
        self.assertEqual(self.score(), 24)
        self.assertEqual(self.score('triple_captain'), 26)

    def test_triple_captain_passes_to_the_vice(self):
        self.assertEqual(self.score('triple_captain', {**self.MINS, 9: 0}), 20 + 2 + 4)

    def test_bench_boost_counts_the_bench(self):
        self.assertEqual(self.score('bench_boost'), 24 + 8)


if __name__ == '__main__':
    unittest.main()
