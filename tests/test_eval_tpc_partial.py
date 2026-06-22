import tempfile
import unittest
from pathlib import Path

from eval_tpc_partial import existing_result_ids, parse_args


class PartialEvaluationTest(unittest.TestCase):
    def test_existing_results_preserve_split_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "c.json").write_text("{}", encoding="utf-8")
            (path / "a.json").write_text("{}", encoding="utf-8")
            (path / "unrelated.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                ["a", "c"],
                existing_result_ids(["a", "b", "c"], path),
            )

    def test_default_method_matches_runner(self):
        args = parse_args([])
        self.assertEqual(
            "UrbanTrip_TPCLLM_en_oracletranslation", args.method
        )


if __name__ == "__main__":
    unittest.main()
