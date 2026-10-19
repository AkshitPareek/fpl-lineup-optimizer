import contextlib
import io
import unittest
from unittest import mock

import pandas as pd
import pulp

from advanced_optimizer import MultiPeriodFPLOptimizer

TEAMS = 10
SLOTS = [(1, 2), (2, 3), (3, 3), (4, 2)]  # (position, players per team)


def synthetic_league(first_gw, last_gw=38):
    players, predictions = [], []
    pid = 0
    for team in range(1, TEAMS + 1):
        for pos, count in SLOTS:
            for k in range(count):
                pid += 1
                players.append({
                    'id': pid, 'web_name': f'P{pid}', 'element_type': pos, 'team': team,
                    'now_cost': 45 + 5 * k, 'status': 'a', 'selected_by_percent': '5.0',
                })
                row = {'id': pid}
                for gw in range(first_gw, last_gw + 1):
                    row[f'xp_gw{gw}'] = 2.0 + 0.3 * team + 0.5 * k
                predictions.append(row)
    teams = pd.DataFrame({'id': range(1, TEAMS + 1), 'name': [f'T{t}' for t in range(1, TEAMS + 1)],
                          'short_name': [f'T{t}' for t in range(1, TEAMS + 1)]})
    for side in ['attack_home', 'attack_away', 'defence_home', 'defence_away']:
        teams[f'strength_{side}'] = 1100
    players = pd.DataFrame(players)
    return players, teams, players.merge(pd.DataFrame(predictions), on='id')


def plan(current_gw, gameweeks):
    players, teams, predictions = synthetic_league(current_gw)
    optimizer = MultiPeriodFPLOptimizer(players, teams, [], current_gw, predictions_df=predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        return optimizer, optimizer.optimize_multi_period(budget=100.0, gameweeks=gameweeks)


def cheapest_top_squad(players):
    """Fifteen £4.5m players from the five best teams, three per team, so the only upgrades cost money."""
    def pick(pos, team):
        return int(players[(players.element_type == pos) & (players.team == team) & (players.now_cost == 45)].id.iloc[0])
    squad = [pick(1, 6), pick(1, 7)] + [pick(2, t) for t in range(6, 11)] + [pick(3, t) for t in range(6, 11)]
    return squad + [pick(4, t) for t in (8, 9, 10)]


def plan_from_squad(squad_bank, selling=None, **kwargs):
    players, teams, predictions = synthetic_league(5)
    squad = cheapest_top_squad(players)
    selling_prices = {pid: selling for pid in squad} if selling else None
    budget = (selling or 4.5) * len(squad) + squad_bank
    optimizer = MultiPeriodFPLOptimizer(players, teams, [], 5, predictions_df=predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        solution = optimizer.optimize_multi_period(budget=budget, current_squad_ids=squad,
                                                   selling_prices=selling_prices, **kwargs)
    return players, squad, solution


class SellingPriceTests(unittest.TestCase):
    def test_upgrades_are_paid_for_at_selling_prices(self):
        players, _, at_current = plan_from_squad(0.5, gameweeks=3, banked_transfers=1, max_hits=0)
        bought = [t for tp in at_current.transfer_summary for t in tp.transfers_in]
        self.assertTrue(any(t['cost'] >= 5.0 for t in bought))

        # Bought at £4.3m and now £4.5m, each sells for £4.4m, leaving every upgrade £0.1m short.
        _, squad, at_selling = plan_from_squad(0.5, selling=4.4, gameweeks=3, banked_transfers=1, max_hits=0)
        self.assertEqual(at_selling.status, 'Optimal')
        bought = [t for tp in at_selling.transfer_summary for t in tp.transfers_in]
        self.assertFalse(any(t['cost'] >= 5.0 for t in bought))
        for tp in at_selling.transfer_summary:
            for sold in tp.transfers_out:
                # Players bought during the plan sell for what they cost.
                self.assertEqual(sold['cost'], 4.4 if sold['id'] in squad else 4.5)
        self.assertTrue(all(gp.bank >= 0 for gp in at_selling.gameweek_plans))

    def test_a_forecast_rise_is_charged_on_the_buy(self):
        players, teams, predictions = synthetic_league(5)
        target = int(players[(players.element_type == 3) & (players.now_cost == 50)].id.iloc[0])
        players.loc[players.id == target, 'selected_by_percent'] = '20.0'
        players.loc[players.id == target, 'transfers_in_event'] = 500000
        players.loc[players.id == target, 'transfers_out_event'] = 50000
        players.loc[players.id == target, 'cost_change_event'] = 0
        predictions.loc[predictions.id == target, 'xp_gw5'] = 20
        # 20% of 10m owners, 450k net transfers: three rises, so 5.0 becomes 5.3.
        # Bank of 0.6 buys at 5.0 (selling a 4.5) and does not buy at 5.3.
        optimizer = MultiPeriodFPLOptimizer(players, teams, [], 5, predictions_df=predictions)
        squad = cheapest_top_squad(players)
        with contextlib.redirect_stdout(io.StringIO()):
            short = optimizer.optimize_multi_period(
                budget=4.5 * len(squad) + 0.6, gameweeks=3, current_squad_ids=squad,
                banked_transfers=1, max_hits=0, total_managers=10_000_000, transfer_elapsed=1)
        bought = [t['id'] for tp in short.transfer_summary for t in tp.transfers_in]
        self.assertNotIn(target, bought)

        with contextlib.redirect_stdout(io.StringIO()):
            funded = optimizer.optimize_multi_period(
                budget=4.5 * len(squad) + 1.0, gameweeks=3, current_squad_ids=squad,
                banked_transfers=1, max_hits=0, total_managers=10_000_000, transfer_elapsed=1)
        ins = [t for tp in funded.transfer_summary for t in tp.transfers_in if t['id'] == target]
        self.assertEqual(len(ins), 1)
        self.assertAlmostEqual(ins[0]['cost'], 5.3)

    def test_bank_follows_the_transfers(self):
        _, _, solution = plan_from_squad(1.5, selling=4.4, gameweeks=4, banked_transfers=2, max_hits=0)
        bank = 1.5
        for gp in solution.gameweek_plans:
            bank += sum(t['cost'] for t in gp.transfers.transfers_out) - sum(t['cost'] for t in gp.transfers.transfers_in)
            self.assertAlmostEqual(gp.bank, bank, places=6)


class ChipTransferTests(unittest.TestCase):
    def test_free_hit_squad_reverts_the_next_week(self):
        _, squad, solution = plan_from_squad(5.0, gameweeks=3, banked_transfers=0, max_hits=0,
                                             chip_to_use=('free_hit', 6))
        weeks = {gp.gameweek: {p['id'] for p in gp.starting_xi + gp.bench} for gp in solution.gameweek_plans}
        self.assertEqual(weeks[5], set(squad))
        self.assertGreater(len(weeks[6] - set(squad)), 1)
        # The free transfer earned in GW5 is kept through the Free Hit, so GW7 can make one move.
        self.assertLessEqual(len(weeks[7] - set(squad)), 1)

    def test_wildcard_week_keeps_but_does_not_add_a_free_transfer(self):
        _, _, solution = plan_from_squad(5.0, gameweeks=3, banked_transfers=2, max_hits=0,
                                         chip_to_use=('wildcard', 5))
        later_moves = sum(gp.transfers.free_transfers_used for gp in solution.gameweek_plans[1:])
        # Two kept through the GW5 wildcard, plus one after each of GW6 and GW7.
        self.assertEqual(solution.banked_transfers_after, min(5, 2 + 2 - later_moves))


class LongHorizonTests(unittest.TestCase):
    def test_plans_up_to_fifteen_gameweeks(self):
        _, solution = plan(current_gw=5, gameweeks=15)
        self.assertEqual(solution.status, 'Optimal')
        self.assertEqual([p.gameweek for p in solution.gameweek_plans], list(range(5, 20)))
        for gw_plan in solution.gameweek_plans:
            self.assertEqual((len(gw_plan.starting_xi), len(gw_plan.bench)), (11, 4))
            self.assertIn(gw_plan.captain_id, [p['id'] for p in gw_plan.starting_xi])
            others = [p for p in gw_plan.starting_xi if p['id'] != gw_plan.captain_id]
            vice = next(p for p in others if p['id'] == gw_plan.vice_captain_id)
            self.assertEqual(vice['expected_points'], max(p['expected_points'] for p in others))

    def test_lineups_survive_solver_tolerance(self):
        class NearIntegerSolver(pulp.PULP_CBC_CMD):
            def actualSolve(self, lp, **kwargs):
                status = super().actualSolve(lp, **kwargs)
                for var in lp.variables():
                    if var.varValue and var.varValue > 0.5:
                        var.varValue -= 1e-7
                return status

        plain = staticmethod(lambda: pulp.PULP_CBC_CMD(msg=0))
        nudging = staticmethod(lambda: NearIntegerSolver(msg=0))
        with mock.patch.object(MultiPeriodFPLOptimizer, '_multi_period_solver', plain):
            _, exact = plan(current_gw=5, gameweeks=4)
        with mock.patch.object(MultiPeriodFPLOptimizer, '_multi_period_solver', nudging):
            _, nudged = plan(current_gw=5, gameweeks=4)
        for gw_plan in nudged.gameweek_plans:
            self.assertEqual((len(gw_plan.starting_xi), len(gw_plan.bench)), (11, 4))
            self.assertIsNotNone(gw_plan.captain_id)
        self.assertEqual(nudged.banked_transfers_after, exact.banked_transfers_after)

    def test_near_integer_solver_values_are_read_as_whole(self):
        binary = pulp.LpVariable('picked', cat='Binary')
        binary.varValue = 0.9999999
        self.assertTrue(MultiPeriodFPLOptimizer._chosen(binary))
        binary.varValue = 1e-7
        self.assertFalse(MultiPeriodFPLOptimizer._chosen(binary))
        count = pulp.LpVariable('hits', cat='Integer')
        count.varValue = 1.9999998
        self.assertEqual(MultiPeriodFPLOptimizer._count(count), 2)

    def test_longer_requests_are_capped(self):
        _, solution = plan(current_gw=5, gameweeks=30)
        self.assertEqual(len(solution.gameweek_plans), MultiPeriodFPLOptimizer.MAX_HORIZON)

    def test_plan_stops_at_the_last_gameweek(self):
        _, solution = plan(current_gw=33, gameweeks=15)
        self.assertEqual([p.gameweek for p in solution.gameweek_plans], list(range(33, 39)))
        _, solution = plan(current_gw=37, gameweeks=8)
        self.assertEqual([p.gameweek for p in solution.gameweek_plans], [37, 38])

    def test_solver_stops_within_a_twentieth_of_a_point(self):
        solver = MultiPeriodFPLOptimizer._multi_period_solver()
        self.assertAlmostEqual(solver.gapAbs, MultiPeriodFPLOptimizer.SOLVER_ABS_GAP)
        self.assertLessEqual(solver.gapRel, 1e-4)
        self.assertAlmostEqual(MultiPeriodFPLOptimizer.SOLVER_ABS_GAP, 0.05)

    def test_response_says_whether_the_plan_is_proven(self):
        optimizer, solution = plan(current_gw=5, gameweeks=3)
        self.assertTrue(solution.proven_optimal)
        self.assertIs(optimizer.to_dict(solution)['proven_optimal'], True)


if __name__ == '__main__':
    unittest.main()
