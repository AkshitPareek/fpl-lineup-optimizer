import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import predictor_eval as pe


class DeadlineFlagsTests(unittest.TestCase):
    def flags(self, season, rows):
        folder = tempfile.mkdtemp()
        pd.DataFrame(rows).to_csv(os.path.join(folder, f"{season}.csv"), index=False)
        with mock.patch.object(pe, "FLAGS_DIR", folder):
            return pe.deadline_flags(season)

    def test_a_snapshot_taken_during_a_gameweek_is_the_next_deadlines_flags(self):
        flags = self.flags("2024-25", [
            {"id": 7, "gw": 10, "status": "d", "chance_of_playing_next_round": 75},
        ])
        self.assertEqual(flags.loc[(11, 7), "status"], "d")
        self.assertEqual(flags.loc[(11, 7), "news"], "")  # 2024/25 has no news column

    def test_early_2025_26_flags_are_left_out(self):
        flags = self.flags("2025-26", [
            {"id": 7, "gw": 9, "status": "i", "chance_of_playing_next_round": 0, "news": "Knee injury"},
            {"id": 7, "gw": 11, "status": "i", "chance_of_playing_next_round": 0, "news": "Knee injury"},
        ])
        self.assertEqual(list(flags.index), [(12, 7)])

    def test_seasons_without_flags_have_none(self):
        self.assertIsNone(pe.deadline_flags("2023-24"))


if __name__ == "__main__":
    unittest.main()
