"""Prepared integration regressions with invented rows and mocked caller. NOT RUN."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
import pandas as pd
from run_contract import ReasoningConfig
SPEC = importlib.util.spec_from_file_location("knhib_pipeline", Path(__file__).parents[1] / "code/benchmark_pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class PipelineIdentity(unittest.TestCase):
    def fixture(self, expected="eligible"):
        return dict(ID="SYNTHETIC-1", cancer_type="invented", regimen="invented", attributes="invented",
                    expected=expected, **{"class":"pos"})

    def test_gold_is_not_in_model_prompt(self):
        self.assertEqual(pipeline.build_forward_user_prompt(self.fixture()),
                         pipeline.build_forward_user_prompt(self.fixture("ineligible")))

    def test_mixed_arm_retry_selects_exact_effort_before_transport(self):
        rows = [dict(id="SYNTHETIC-1", model="gpt-6-luna", condition="pdf", predicted="error", run_id=0,
                     episode_id="medium", reasoning_effort="medium", reasoning_mode="standard", experiment_id="x",
                     input_hash="h", timestamp="2026-10-20T00:00:00", gold_hash="g", fixture_hash="f"),
                dict(id="SYNTHETIC-1", model="gpt-6-luna", condition="pdf", predicted="error", run_id=0,
                     episode_id="max", reasoning_effort="max", reasoning_mode="standard", experiment_id="x",
                     input_hash="h", timestamp="2026-10-20T00:00:00", gold_hash="g", fixture_hash="f")]
        plan = pipeline.build_retry_task_plan(pd.DataFrame([self.fixture()]), pd.DataFrame(rows),
                ["gpt-6-luna"], ["pdf"], ReasoningConfig("medium"))
        self.assertEqual(plan["episode_id"].tolist(), ["medium"])

    def test_get_pdf_text_has_no_output_side_effect(self):
        parser = Mock()
        parser.extract_text.return_value="Invented criterion"
        with patch.dict(sys.modules, {"pdf_parser":parser}), patch.object(Path,"mkdir",side_effect=AssertionError("outside output write")):
            self.assertEqual(pipeline.get_pdf_text("synthetic-unique.pdf"), "Invented criterion")
