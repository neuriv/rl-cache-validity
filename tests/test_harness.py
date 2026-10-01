import json
import tempfile
import unittest
from pathlib import Path

from rl_cache_validity.harness import main, run


class HarnessTests(unittest.TestCase):
    def test_controls_and_chunk_floor(self):
        result = run(seed=71, arches=("llama",))
        rows = {row["case"]: row for row in result["models"]}
        for case in ("zero_update", "head_only", "null_history"):
            self.assertTrue(rows[case]["first_layer_B_cache_equal_R_F"])
        self.assertEqual(rows["zero_update"]["kl_R_to_F"], 0.0)
        self.assertEqual(rows["head_only"]["cache_diff_R_F"]["max_abs"], 0.0)
        self.assertTrue(rows["head_only"]["head_only_untied_embedding_unchanged"])
        self.assertGreater(rows["null_history"]["kl_R_to_fresh_theta2"], 0.0)
        self.assertTrue(rows["history"]["deeper_B_cache_differs_R_F"])
        self.assertEqual(rows["history"]["cache_length"], 5)
        self.assertGreaterEqual(rows["history"]["kl_segmented_fresh_to_monolithic_fresh"], 0.0)

    def test_cli_result_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.json", Path(directory) / "second.json"
            for output in (first, second):
                main(["--architecture", "llama", "--seed", "81", "--output", str(output)])
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(json.loads(first.read_text())["seed"], 81)


if __name__ == "__main__":
    unittest.main()
