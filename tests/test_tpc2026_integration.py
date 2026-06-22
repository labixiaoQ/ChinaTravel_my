import argparse
import unittest

from agent_env.adapter import ChinaTravelEnvAdapter
from chinatravel.data.load_datasets import load_query


SPLIT = "TPC_IJCAI_2026_phase1"
FIRST_UID = "20250320174446059265"


class TPC2026IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.query_ids, cls.query_data = load_query(
            argparse.Namespace(
                splits=SPLIT,
                lang="en",
                oracle_translation=True,
            )
        )

    def test_phase1_index_loads_all_english_queries(self):
        self.assertEqual(1000, len(self.query_ids))
        self.assertEqual(FIRST_UID, self.query_ids[0])
        self.assertEqual(self.query_ids, list(self.query_data))
        self.assertEqual("Shanghai", self.query_data[FIRST_UID]["start_city"])
        self.assertIn("hard_logic_py", self.query_data[FIRST_UID])

    def test_legacy_split_alias_targets_tpc2026_data(self):
        query_ids, query_data = load_query(
            argparse.Namespace(
                splits="tpc_phase1",
                lang="en",
                oracle_translation=True,
            )
        )
        self.assertEqual(self.query_ids, query_ids)
        self.assertEqual(self.query_data[FIRST_UID], query_data[FIRST_UID])

    def test_agent_adapter_returns_query_with_hard_logic(self):
        adapter = ChinaTravelEnvAdapter(lang="en")
        result = adapter.load_query(SPLIT, FIRST_UID)
        self.assertTrue(result["success"], result)
        self.assertEqual("en", result["lang"])
        self.assertIn("hard_logic_py", result["query"])

    def test_agent_adapter_queries_english_database(self):
        adapter = ChinaTravelEnvAdapter(lang="en")
        self.assertEqual("en", adapter._get_env().lang)
        result = adapter.call_tool("attractions_keys", {"city": "Chengdu"})
        self.assertTrue(result["success"], result)
        self.assertIn("name", result["text"].lower())


if __name__ == "__main__":
    unittest.main()
