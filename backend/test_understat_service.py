import tempfile
import unittest
from unittest import mock

import understat_service as us


class LeagueFetchTests(unittest.TestCase):
    def service(self):
        with mock.patch.object(us.UnderstatService, "CACHE_DIR", tempfile.mkdtemp()):
            return us.UnderstatService()

    def test_a_failed_fetch_is_not_retried_by_the_next_plan(self):
        service = self.service()
        client = mock.MagicMock()
        client.return_value.__enter__.return_value.league.side_effect = ConnectionError("timed out")
        with mock.patch.object(us, "UnderstatClient", client, create=True), \
                mock.patch.object(us, "UNDERSTAT_AVAILABLE", True):
            self.assertEqual(service.get_league_players("2026"), [])
            self.assertEqual(service.get_league_players("2026"), [])
        self.assertEqual(client.call_count, 1)


if __name__ == "__main__":
    unittest.main()
