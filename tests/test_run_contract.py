"""Offline, invented fixtures only. Prepared; execution requires the approved batch."""
import importlib.util
from pathlib import Path
import sys
import unittest

SPEC = importlib.util.spec_from_file_location("knhib_run_contract", Path(__file__).parents[1] / "code/run_contract.py")
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


class RequestContracts(unittest.TestCase):
    def request(self, **overrides):
        kwargs = dict(model="gpt-6-luna", instructions="Invented test only",
                      user_prompt="Invented administrative facts", condition="text_md",
                      config=contract.ReasoningConfig("medium"), guideline_text="Invented criterion")
        kwargs.update(overrides)
        return contract.build_request(**kwargs)

    def test_exact_efforts_and_mode(self):
        for effort in ("medium", "max"):
            p = self.request(config=contract.ReasoningConfig(effort, "standard", 256))
            self.assertEqual(p["model"], "gpt-6-luna")
            self.assertEqual(p["reasoning"], {"effort": effort, "mode": "standard"})
            self.assertEqual(p["max_output_tokens"], 256)
            self.assertFalse(set(p) & {"temperature", "top_p", "top_logprobs"})

    def test_unknown_effort_and_implicit_luna_fail(self):
        for effort in (None, "high", "xhigh", "MAX"):
            with self.assertRaises(ValueError):
                self.request(config=contract.ReasoningConfig(effort))

    def test_text_condition_includes_guideline_and_search(self):
        p = self.request(condition="text_md_websearch", model="gpt-5.4",
                         config=contract.ReasoningConfig())
        self.assertIn("Invented criterion", p["input"][0]["content"][0]["text"])
        self.assertEqual(p["tools"], [{"type": "web_search"}])
        self.assertNotIn("tools", self.request())

    def test_missing_guidelines_fail(self):
        with self.assertRaises(ValueError):
            self.request(guideline_text="")
        with self.assertRaises(ValueError):
            self.request(condition="pdf")

    def test_no_gold_leak(self):
        p = self.request()
        self.assertNotIn("expected", str(p))

    def test_usage_unknown_is_not_zero(self):
        m = contract.response_metadata({"model":"gpt-6-luna", "status":"completed"}, "gpt-6-luna")
        self.assertIsNone(m["input_tokens"])
        self.assertEqual(m["usage_status"], "unknown")

    def test_substitution_and_incomplete_fail(self):
        for response in ({"model":"gpt-5.4", "status":"completed"},
                         {"model":"gpt-6-luna", "status":"incomplete"}):
            with self.assertRaises(ValueError):
                contract.response_metadata(response, "gpt-6-luna")

    def test_episode_replication_and_effort_identity(self):
        kw = dict(case_id="SYNTHETIC-1", model="gpt-6-luna", condition="text_md", run_id=0,
                  effort="medium", mode="standard", experiment_id="offline-fixture", input_hash="invented")
        first = contract.episode_id(**kw)
        self.assertNotEqual(first, contract.episode_id(**dict(kw, effort="max")))
        self.assertNotEqual(first, contract.episode_id(**dict(kw, run_id=1)))
        self.assertNotEqual(first, contract.episode_id(**dict(kw, input_hash="changed")))

    def test_case_selection_stable_and_exact(self):
        self.assertEqual(contract.selected_ids(["S-1", "S-2", "S-3"], ["S-3", "S-1"]), ["S-1", "S-3"])
        for requested in (["unknown"], ["S-1", "S-1"]):
            with self.assertRaises(ValueError):
                contract.selected_ids(["S-1"], requested)


if __name__ == "__main__":
    unittest.main()
