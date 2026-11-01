import contextlib
import io
import unittest
from unittest import mock

import requests
from fastapi.testclient import TestClient

from advanced_optimizer import MultiPeriodFPLOptimizer
from fpl_service import FPLService
from test_long_horizon import cheapest_top_squad, synthetic_league


def setUpModule():
    # Made-up players never match Understat, so do not ask it.
    patcher = mock.patch('point_predictor.UNDERSTAT_AVAILABLE', False)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def plan(**kwargs):
    """Four weeks from a squad of cheap players with £10m to upgrade, so the plan sells some of them."""
    players, teams, predictions = synthetic_league(5)
    squad = cheapest_top_squad(players)
    optimizer = MultiPeriodFPLOptimizer(players, teams, [], 5, predictions_df=predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        solution = optimizer.optimize_multi_period(
            budget=4.5 * len(squad) + 10.0, gameweeks=4, current_squad_ids=squad, banked_transfers=2, **kwargs)
    return players, squad, solution


def squads(solution):
    return [{p['id'] for p in gp.starting_xi + gp.bench} for gp in solution.gameweek_plans]


class LockedPlayerTests(unittest.TestCase):
    def test_a_kept_player_the_plan_would_sell_stays_every_week(self):
        _, squad, free = plan()
        sold = next(t['id'] for gp in free.gameweek_plans for t in gp.transfers.transfers_out)
        _, _, kept = plan(locked_players=[sold])
        self.assertEqual(kept.status, 'Optimal')
        self.assertTrue(all(sold in week for week in squads(kept)))

    def test_a_kept_player_not_owned_is_bought_in_the_first_week(self):
        players, squad, free = plan()
        wanted = int(players[(players.element_type == 3) & (players.team == 1)].id.iloc[0])
        self.assertFalse(any(wanted in week for week in squads(free)))
        _, _, kept = plan(locked_players=[wanted])
        self.assertTrue(all(wanted in week for week in squads(kept)))

    def test_a_player_both_kept_and_banned_is_banned(self):
        _, squad, _ = plan()
        _, _, solution = plan(locked_players=[squad[0]], excluded_players=[squad[0]])
        self.assertTrue(all(squad[0] not in week for week in squads(solution)))


class ChipReasonTests(unittest.TestCase):
    def test_a_planned_triple_captain_reads_as_a_comparison_with_keeping_it(self):
        players, teams, predictions = synthetic_league(5)
        optimizer = MultiPeriodFPLOptimizer(players, teams, [], 5, predictions_df=predictions)

        class Week:
            gameweek = 6
            captain_id = 1
            starting_xi = [{'id': 1, 'expected_points': 7.3}]
            bench = []
            chip_used = mock.Mock(value='3xc')

        windows = [{'chip': 'triple_captain', 'weeks': [5, 6, 7], 'weeks_after': 4}]
        suggestion, _ = optimizer._planned_chip_advice(windows, {'triple_captain': 6.8}, [Week()])
        self.assertEqual(suggestion['reason'],
                         'Triple Captain in GW6 adds about 7.3 points. The best week after the plan would add about 6.8.')


class Response:
    def __init__(self, code, body=None):
        self.status_code = code
        self._body = body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)

    def json(self):
        return self._body


class ManagerEntryTests(unittest.TestCase):
    def test_an_unknown_id_is_reported_as_missing_without_retrying(self):
        calls = []

        def get(url, headers=None, timeout=None):
            calls.append(url)
            return Response(404)

        with mock.patch('fpl_service.requests.get', get), mock.patch('fpl_service.time.sleep'):
            self.assertIsNone(FPLService().get_manager_entry(999999999))
        self.assertEqual(len(calls), 1)

    def test_the_entry_names_the_team_and_its_manager(self):
        body = {'name': 'krAne', 'player_first_name': 'Akshit', 'player_last_name': 'Pareek',
                'summary_overall_rank': 4039613, 'summary_overall_points': 508}
        with mock.patch('fpl_service.requests.get', return_value=Response(200, body)):
            entry = FPLService().get_manager_entry(7706676)
        self.assertEqual(entry, {'name': 'krAne', 'player_name': 'Akshit Pareek',
                                 'overall_rank': 4039613, 'overall_points': 508})

    def test_the_account_endpoint_answers_404_for_an_unknown_id(self):
        import main
        client = TestClient(main.app, raise_server_exceptions=False)
        with mock.patch.object(main.fpl_service, 'get_manager_entry', return_value=None), \
                mock.patch.object(main.fpl_service, 'get_manager_chips') as chips, \
                contextlib.redirect_stdout(io.StringIO()):
            response = client.get('/api/manager-chips/999999999')
        self.assertEqual(response.status_code, 404)
        chips.assert_not_called()

    def test_the_account_endpoint_still_loads_when_the_entry_cannot_be_read(self):
        import main
        client = TestClient(main.app, raise_server_exceptions=False)
        with mock.patch.object(main.fpl_service, 'get_manager_entry', side_effect=requests.ConnectionError()), \
                mock.patch.object(main.fpl_service, 'get_manager_chips', return_value={'synced': True}), \
                contextlib.redirect_stdout(io.StringIO()):
            response = client.get('/api/manager-chips/7706676')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'synced': True, 'entry': None})


class TrackRecordTests(unittest.TestCase):
    def test_finished_weeks_are_scored_and_open_ones_only_listed(self):
        import json, os, tempfile
        import scorecard
        player = lambda pid, pos: {'id': pid, 'name': f'P{pid}', 'element_type': pos, 'xp': 5.0}
        xi = [player(1, 1)] + [player(i, 2) for i in range(2, 6)] + [player(i, 3) for i in range(6, 10)] + [player(10, 4), player(11, 4)]
        entry = {'gameweek': 8, 'expected_points': 60.0, 'starting_xi': xi, 'bench': [player(12, 1)],
                 'captain_id': 10, 'vice_captain_id': 11, 'chip': None, 'transfers_in': ['P10'],
                 'transfers_out': ['P99'], 'hits': 0}
        api = {
            'bootstrap-static/': {'events': [{'id': 8, 'finished': True, 'average_entry_score': 45, 'highest_score': 110},
                                             {'id': 9, 'finished': False}]},
            'entry/5/history/': {'current': [{'event': 8, 'points': 50, 'event_transfers_cost': 4}]},
            'event/8/live/': {'elements': [{'id': i, 'stats': {'total_points': 2, 'minutes': 90}} for i in range(1, 13)]},
        }
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, '5'))
            for gw in (8, 9):
                with open(os.path.join(root, '5', f'gw{gw:02d}.json'), 'w') as f:
                    json.dump({**entry, 'gameweek': gw}, f)
            with mock.patch.object(scorecard, 'RECORDS_DIR', root):
                weeks = scorecard.scored_weeks(5, api.__getitem__)
        self.assertEqual([w['gameweek'] for w in weeks], [8, 9])
        # Eleven starters at 2 points and the captain's 2 again
        self.assertEqual((weeks[0]['advice'], weeks[0]['you'], weeks[0]['average']), (24, 46, 45))
        self.assertEqual(weeks[0]['captain'], 'P10')
        self.assertFalse(weeks[1]['finished'])
        self.assertNotIn('advice', weeks[1])


if __name__ == '__main__':
    unittest.main()
