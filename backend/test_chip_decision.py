import unittest

from chip_advisor import choose_chip_to_play


def tc(gw, xp, dgw=False):
    return {
        'chip': 'triple_captain',
        'recommended_gameweek': gw,
        'estimated_gain': xp,
        'is_dgw': dgw,
        'reason': f'captain {xp}',
    }


class ChipDecisionTests(unittest.TestCase):
    def test_a_flat_captain_week_is_not_a_triple_captain(self):
        decision = choose_chip_to_play([tc(8, 7.5), tc(9, 7.4), tc(10, 7.2)])
        self.assertIsNone(decision['play'])
        self.assertEqual(decision['held'][0]['chip'], 'triple_captain')
        self.assertIn('Hold Triple Captain', decision['held'][0]['reason'])

    def test_a_double_gameweek_spike_plays_the_chip(self):
        decision = choose_chip_to_play([tc(12, 14.0, dgw=True), tc(13, 7.2), tc(14, 7.0)])
        self.assertEqual(decision['play']['recommended_gameweek'], 12)
        self.assertTrue(decision['play']['play'])
        self.assertEqual(decision['held'], [])

    def test_one_week_is_not_enough_to_spend_the_chip(self):
        decision = choose_chip_to_play([tc(9, 8.0)])
        self.assertIsNone(decision['play'])
        self.assertIn('One week is not enough', decision['held'][0]['reason'])


if __name__ == '__main__':
    unittest.main()
