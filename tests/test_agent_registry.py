import unittest

from chinatravel.agent.registry import agent_names, get_agent_spec
from run_tpc2026 import method_name, parse_args, select_query_ids


class AgentRegistryTest(unittest.TestCase):
    def test_all_legacy_agents_are_registered(self):
        self.assertEqual(
            {
                "RuleNeSy", "LLMNeSy", "LLM-modulo", "ReAct", "ReAct0",
                "Act", "TPCAgent", "UrbanTrip",
            },
            set(agent_names()),
        )

    def test_tpc2026_defaults_to_urbantrip_english_and_hard_logic(self):
        args = parse_args([])
        self.assertEqual("UrbanTrip", args.agent)
        self.assertEqual("en", args.lang)
        self.assertTrue(args.oracle_translation)
        self.assertEqual("TPCLLM", get_agent_spec(args.agent).default_llm)

    def test_query_selection_validates_uid_and_limit(self):
        ids = ["a", "b", "c"]
        self.assertEqual(["a", "b"], select_query_ids(ids, limit=2))
        self.assertEqual(["b"], select_query_ids(ids, index="b"))
        with self.assertRaises(ValueError):
            select_query_ids(ids, index="missing")

    def test_method_name_is_evaluation_compatible(self):
        args = parse_args([])
        self.assertEqual(
            "UrbanTrip_TPCLLM_en_oracletranslation",
            method_name(args, "TPCLLM"),
        )


if __name__ == "__main__":
    unittest.main()
