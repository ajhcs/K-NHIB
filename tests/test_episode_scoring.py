"""Prepared pandas/scorer regression tests. No API, workbook or guideline access."""
import importlib.util
from pathlib import Path
import unittest
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
import pandas as pd

SPEC = importlib.util.spec_from_file_location("knhib_scoring", Path(__file__).parents[1] / "code/aggregate_results.py")
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


class EpisodeScoring(unittest.TestCase):
    def row(self, **kw):
        d = dict(episode_id="e1", attempt_id="a1", experiment_id="invented", run_id=0,
                 reasoning_effort="medium", reasoning_mode="standard", input_hash="fixture", configuration_id="config", fixture_hash="synthetic-fixture", gold_hash="synthetic-gold", is_dry_run=False,
                 timestamp="2026-10-20T00:00:00Z", predicted="error", expected="eligible")
        d.update(kw)
        return d

    def test_efforts_and_replicates_survive(self):
        rows = [self.row(), self.row(episode_id="e2", attempt_id="a2", reasoning_effort="max"),
                self.row(episode_id="e3", attempt_id="a3", run_id=1)]
        self.assertEqual(len(scorer.deduplicate(pd.DataFrame(rows))), 3)

    def test_retry_keeps_latest_without_hiding_other_episodes(self):
        rows = [self.row(), self.row(attempt_id="a2", timestamp="2026-10-20T00:01:00Z", predicted="eligible")]
        final = scorer.deduplicate(pd.DataFrame(rows))
        self.assertEqual(final.iloc[0]["attempt_id"], "a2")
        self.assertEqual(len(rows), 2)

    def test_conflicting_attempt_rejected(self):
        with self.assertRaises(ValueError):
            scorer.deduplicate(pd.DataFrame([self.row(), self.row(predicted="eligible")]))

    def test_missing_identity_rejected(self):
        with self.assertRaises(ValueError):
            scorer.deduplicate(pd.DataFrame([{"id": "old-case"}]))

    def test_errors_remain_in_denominator(self):
        metrics = scorer.compute_metrics(pd.DataFrame([self.row()]))
        self.assertEqual(metrics["n"], 1)
        self.assertEqual(metrics["accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
