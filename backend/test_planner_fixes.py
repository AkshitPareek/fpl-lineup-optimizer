import unittest

from fpl_service import banked_transfers_for_next_deadline
from main import _comparison_entry


class BankedTransferTests(unittest.TestCase):
    def test_accrues_unused_transfers_and_spends_them(self):
        events = [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 0},
            {"event": 3, "event_transfers": 0},
            {"event": 4, "event_transfers": 2},
            {"event": 5, "event_transfers": 2},
        ]
        # GW6 starts with 2 free transfers after this history.
        self.assertEqual(banked_transfers_for_next_deadline(events), 2)

    def test_wildcard_does_not_spend_the_bank(self):
        events = [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 8},
        ]
        chips = [{"event": 2, "name": "wildcard"}]
        # After an unused GW1 the bank is 2. The wildcard spends none, then one more is granted.
        self.assertEqual(banked_transfers_for_next_deadline(events, chips), 3)

    def test_empty_history_starts_with_one(self):
        self.assertEqual(banked_transfers_for_next_deadline([]), 1)


class ComparisonSummaryTests(unittest.TestCase):
    def test_infeasible_plan_is_not_scored_as_zero(self):
        class Solution:
            status = "Infeasible"
            total_expected_points = 0

        summary = _comparison_entry(Solution(), 0, 0, 0, True)
        self.assertFalse(summary["feasible"])
        self.assertIsNone(summary["net_xp"])
        self.assertFalse(summary["recommended"])


if __name__ == "__main__":
    unittest.main()
