import contextlib
import io
import unittest

from advanced_optimizer import MultiPeriodFPLOptimizer
from test_long_horizon import cheapest_top_squad, synthetic_league


def plan_chips(current_gw, gameweeks, adjust=None, chips_used=(), optimize_chips=True, **kwargs):
    """Plan from the cheap top-team squad, after adjust(predictions, squad) edits the projections."""
    players, teams, predictions = synthetic_league(current_gw)
    squad = cheapest_top_squad(players)
    if adjust:
        adjust(predictions, squad)
    optimizer = MultiPeriodFPLOptimizer(players, teams, [], current_gw, predictions_df=predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        solution = optimizer.optimize_multi_period(
            budget=4.5 * len(squad) + kwargs.pop('bank', 1.0), gameweeks=gameweeks,
            current_squad_ids=squad, chips_used=list(chips_used), optimize_chips=optimize_chips,
            **{'banked_transfers': 1, 'max_hits': 0, **kwargs})
    return predictions, squad, solution


def chips_by_week(solution):
    return {gp.gameweek: gp.chip_used.value for gp in solution.gameweek_plans if gp.chip_used}


def captain_spike(gw, points=20.0):
    def adjust(predictions, squad):
        predictions.loc[predictions.id == squad[-1], f'xp_gw{gw}'] = points
    return adjust


def squad_double(gw, extra=5.0):
    def adjust(predictions, squad):
        predictions.loc[predictions.id.isin(squad), f'xp_gw{gw}'] += extra
    return adjust


class AutosubPlanTests(unittest.TestCase):
    def test_a_blanking_starter_makes_a_bench_upgrade_worth_a_hit(self):
        # The bench scores only by coming on. With no free transfer, paying a hit for a
        # substitute is worth it exactly when that substitute's points are in the objective.
        players, teams, predictions = synthetic_league(5)
        for column in predictions.columns:
            if str(column).startswith('xp_'):
                predictions[column] = 0.0
        squad = cheapest_top_squad(players)
        by_pos = {}
        for pid in squad:
            pos = int(players.loc[players.id == pid, 'element_type'].iloc[0])
            by_pos.setdefault(pos, []).append(pid)
        playing = by_pos[1][:1] + by_pos[2][:4] + by_pos[3][:4] + by_pos[4][:2]
        flagged = by_pos[3][0]
        fodder = by_pos[3][4]
        predictions.loc[predictions.id.isin(playing), [c for c in predictions.columns if str(c).startswith('xp_')]] = 10.0
        players.loc[players.id == flagged, 'chance_of_playing_next_round'] = 0
        target = int(players[(players.element_type == 3) & (~players.id.isin(squad)) & (players.now_cost == 45)].id.iloc[0])
        predictions.loc[predictions.id == target, 'xp_gw5'] = 9.0
        optimizer = MultiPeriodFPLOptimizer(players, teams, [], 5, predictions_df=predictions)
        with contextlib.redirect_stdout(io.StringIO()):
            solution = optimizer.optimize_multi_period(
                budget=4.5 * len(squad), gameweeks=3, current_squad_ids=squad,
                banked_transfers=0, max_hits=1, optimize_chips=False)
        self.assertEqual(solution.status, 'Optimal')
        week = next(gp for gp in solution.gameweek_plans if gp.gameweek == 5)
        self.assertEqual([p['id'] for p in week.transfers.transfers_in], [target])
        self.assertEqual([p['id'] for p in week.transfers.transfers_out], [fodder])
        self.assertEqual(week.transfers.hits_taken, 1)
        # 11 players at 10 with one captained, plus the substitute's 9. The hit is paid
        # because those 9 points are in the objective; starting the new player and
        # benching a 10 is worth the same, since the benched player comes on.
        self.assertAlmostEqual(week.expected_points, 129.0, places=4)

    def test_two_certain_blanks_value_two_substitutes(self):
        # Two starters who will not play should bring on two substitutes. Each hit costs 4,
        # so the second substitute is bought only when the objective pays for a second blank.
        players, teams, predictions = synthetic_league(5)
        for column in predictions.columns:
            if str(column).startswith('xp_'):
                predictions[column] = 0.0
        squad = cheapest_top_squad(players)
        by_pos = {}
        for pid in squad:
            pos = int(players.loc[players.id == pid, 'element_type'].iloc[0])
            by_pos.setdefault(pos, []).append(pid)
        playing = by_pos[1][:1] + by_pos[2][:4] + by_pos[3][:3] + by_pos[4][:2]
        flagged = by_pos[3][:2]
        fodder = by_pos[3][3:]
        predictions.loc[predictions.id.isin(playing), [c for c in predictions.columns if str(c).startswith('xp_')]] = 10.0
        players.loc[players.id.isin(flagged), 'chance_of_playing_next_round'] = 0
        outside = players[(players.element_type == 3) & (~players.id.isin(squad)) & (players.now_cost == 45)]
        targets = [int(i) for i in outside.id.iloc[:2]]
        predictions.loc[predictions.id.isin(targets), 'xp_gw5'] = 9.0
        optimizer = MultiPeriodFPLOptimizer(players, teams, [], 5, predictions_df=predictions)
        with contextlib.redirect_stdout(io.StringIO()):
            solution = optimizer.optimize_multi_period(
                budget=4.5 * len(squad), gameweeks=3, current_squad_ids=squad,
                banked_transfers=0, max_hits=2, optimize_chips=False)
        self.assertEqual(solution.status, 'Optimal')
        week = next(gp for gp in solution.gameweek_plans if gp.gameweek == 5)
        self.assertCountEqual([p['id'] for p in week.transfers.transfers_in], targets)
        self.assertCountEqual([p['id'] for p in week.transfers.transfers_out], fodder)
        self.assertEqual(week.transfers.hits_taken, 2)
        # Ten players score 10 and the captain adds 10. The two buys score 9 each from the
        # bench, with a zero-point player filling the last place in the eleven: 128.
        self.assertAlmostEqual(week.expected_points, 128.0, places=4)


class ChipTimingTests(unittest.TestCase):
    def test_triple_captain_goes_on_the_spike_week(self):
        _, squad, solution = plan_chips(5, 4, captain_spike(7))
        self.assertEqual(solution.status, 'Optimal')
        self.assertEqual(chips_by_week(solution), {7: 'triple_captain'})
        week = next(gp for gp in solution.gameweek_plans if gp.gameweek == 7)
        self.assertEqual(week.captain_id, squad[-1])
        self.assertEqual(solution.chip_plan, [
            {'chip': 'triple_captain', 'gameweek': 7, 'expected_gain': 20.0, 'reason': 'P' + str(squad[-1]) + ' projected 20.0 pts'}])

    def test_triple_captain_adds_the_captain_once_more(self):
        _, _, without = plan_chips(5, 4, captain_spike(7), optimize_chips=False)
        _, _, with_chips = plan_chips(5, 4, captain_spike(7))
        self.assertAlmostEqual(with_chips.total_expected_points - without.total_expected_points, 20.0, places=4)

    def test_bench_boost_goes_on_the_week_the_whole_squad_doubles(self):
        _, _, solution = plan_chips(5, 4, squad_double(6))
        self.assertEqual(chips_by_week(solution).get(6), 'bench_boost')
        week = next(gp for gp in solution.gameweek_plans if gp.gameweek == 6)
        self.assertEqual(len(week.bench), 4)

    def test_free_hit_covers_a_blank_and_the_squad_comes_back(self):
        def blank(predictions, squad):
            predictions.loc[predictions.id.isin(squad), 'xp_gw6'] = 0.0
        _, squad, solution = plan_chips(5, 3, blank, bank=5.0)
        self.assertEqual(chips_by_week(solution).get(6), 'freehit')
        weeks = {gp.gameweek: {p['id'] for p in gp.starting_xi + gp.bench} for gp in solution.gameweek_plans}
        # Blanked players score nothing, so none of them start. With no blanks to cover, the
        # bench's own points are not counted and the solver may keep cheap squad players there.
        starters = {p['id'] for p in next(gp for gp in solution.gameweek_plans if gp.gameweek == 6).starting_xi}
        self.assertEqual(len(starters & set(squad)), 0)
        after = next(gp for gp in solution.gameweek_plans if gp.gameweek == 7).transfers
        self.assertEqual(weeks[7], weeks[5] - {p['id'] for p in after.transfers_out}
                         | {p['id'] for p in after.transfers_in})
        free_hit_week = next(gp for gp in solution.gameweek_plans if gp.gameweek == 6)
        self.assertEqual({p['id'] for p in free_hit_week.transfers.transfers_in}, weeks[6] - weeks[5])
        self.assertEqual(free_hit_week.transfers.hits_taken, 0)
        self.assertEqual({p['id'] for p in solution.squad}, weeks[7])

    def test_chip_is_kept_for_a_better_week_after_the_horizon(self):
        _, _, solution = plan_chips(5, 3, captain_spike(12))
        self.assertNotIn('triple_captain', chips_by_week(solution).values())
        held = {h['chip']: h for h in solution.chips_held}
        self.assertEqual(held['triple_captain']['best_gameweek'], 12)
        self.assertAlmostEqual(held['triple_captain']['expected_gain'], 20.0)

    def test_chip_is_played_when_the_better_week_is_inside_the_horizon(self):
        _, _, solution = plan_chips(5, 8, captain_spike(12))
        self.assertEqual(chips_by_week(solution).get(12), 'triple_captain')

    def test_one_chip_a_week(self):
        def both(predictions, squad):
            squad_double(6)(predictions, squad)
            captain_spike(6, 25.0)(predictions, squad)
        _, _, solution = plan_chips(5, 3, both)
        weeks = [p['gameweek'] for p in solution.chip_plan]
        self.assertEqual(len(weeks), len(set(weeks)))
        self.assertIn(6, weeks)

    def test_used_chip_waits_for_the_second_half(self):
        def benches(predictions, squad):
            squad_double(18)(predictions, squad)
            squad_double(21, 3.0)(predictions, squad)
        _, _, solution = plan_chips(17, 5, benches, chips_used=['bboost'])
        self.assertEqual(chips_by_week(solution).get(21), 'bench_boost')
        self.assertNotEqual(chips_by_week(solution).get(18), 'bench_boost')

    def test_each_half_gets_its_own_chip(self):
        def spikes(predictions, squad):
            captain_spike(18)(predictions, squad)
            captain_spike(21)(predictions, squad)
        _, _, solution = plan_chips(17, 5, spikes)
        weeks = {w: c for w, c in chips_by_week(solution).items() if c == 'triple_captain'}
        self.assertEqual(sorted(weeks), [18, 21])

    def test_chips_wait_when_later_weeks_are_as_good(self):
        _, _, solution = plan_chips(5, 3)
        self.assertEqual(chips_by_week(solution), {})

    def test_chips_are_used_before_they_expire(self):
        _, _, solution = plan_chips(15, 5)
        self.assertIn('triple_captain', chips_by_week(solution).values())

    def test_without_optimize_chips_only_the_requested_chip_is_played(self):
        _, _, solution = plan_chips(5, 4, captain_spike(7), optimize_chips=False)
        self.assertEqual(chips_by_week(solution), {})
        _, _, solution = plan_chips(5, 4, captain_spike(7), optimize_chips=False,
                                    chip_to_use=('bench_boost', 6))
        self.assertEqual(chips_by_week(solution), {6: 'bench_boost'})


if __name__ == '__main__':
    unittest.main()
