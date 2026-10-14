import contextlib
import io
import unittest

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


class LongHorizonTests(unittest.TestCase):
    def test_plans_up_to_fifteen_gameweeks(self):
        _, solution = plan(current_gw=5, gameweeks=15)
        self.assertEqual(solution.status, 'Optimal')
        self.assertEqual([p.gameweek for p in solution.gameweek_plans], list(range(5, 20)))
        for gw_plan in solution.gameweek_plans:
            self.assertEqual((len(gw_plan.starting_xi), len(gw_plan.bench)), (11, 4))
            self.assertIn(gw_plan.captain_id, [p['id'] for p in gw_plan.starting_xi])

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

    def test_response_says_whether_the_plan_is_proven(self):
        optimizer, solution = plan(current_gw=5, gameweeks=3)
        self.assertTrue(solution.proven_optimal)
        self.assertIs(optimizer.to_dict(solution)['proven_optimal'], True)


if __name__ == '__main__':
    unittest.main()
