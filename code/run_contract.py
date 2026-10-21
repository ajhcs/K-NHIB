"""Pure request/result contracts. No I/O, credentials, clients or import-time setup."""
from dataclasses import dataclass
import hashlib
import json

LUNA = "gpt-6-luna"


@dataclass(frozen=True)
class ReasoningConfig:
    effort: str | None = None
    mode: str = "standard"
    max_output_tokens: int = 4096
    max_input_tokens: int = 100000

    def validate(self, model: str) -> None:
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if type(self.max_input_tokens) is not int or not 0 < self.max_input_tokens <= 272000:
            raise ValueError("max_input_tokens must be positive and within short-context ceiling")
        if self.mode not in {"standard", "pro"}:
            raise ValueError("Unsupported reasoning mode")
        if model == LUNA and self.effort not in {"medium", "max"}:
            raise ValueError("Luna requires explicit medium or max effort")
        if model != LUNA and (self.effort is not None or self.mode != "standard"):
            raise ValueError("Explicit reasoning settings currently supported only for exact Luna")


def build_request(*, model, instructions, user_prompt, condition, config,
                  guideline_text=None, pdf_data=None, pdf_name=None):
    """Construct exact Responses input; never include gold labels in user_prompt."""
    config.validate(model)
    valid = {"pdf", "pdf_websearch", "pdf_structure", "text_md", "text_md_websearch", "websearch"}
    if model == LUNA and "websearch" in condition:
        raise ValueError("Bounded Luna classification currently excludes billable search tools")
    if condition not in valid:
        raise ValueError("Unknown condition")
    parts = []
    if condition.startswith("pdf"):
        if not pdf_data or not pdf_name:
            raise ValueError("PDF condition requires guideline bytes and filename")
        parts.append({"type": "input_file", "filename": pdf_name,
                      "file_data": "data:application/pdf;base64," + pdf_data})
    elif condition.startswith("text_md"):
        if not isinstance(guideline_text, str) or not guideline_text.strip():
            raise ValueError("Text condition requires nonempty guideline text")
        parts.append({"type": "input_text", "text": "[급여 고시 원문]\n" + guideline_text})
    parts.append({"type": "input_text", "text": user_prompt})
    payload = {"model": model, "instructions": instructions,
               "input": [{"role": "user", "content": parts}],
               "max_output_tokens": config.max_output_tokens}
    if model == LUNA:
        payload["reasoning"] = {"effort": config.effort, "mode": config.mode}
        payload["service_tier"] = "default"
        payload["store"] = False
    if condition in {"websearch", "pdf_websearch", "text_md_websearch"}:
        payload["tools"] = [{"type": "web_search"}]
    return payload


def request_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def episode_id(*, case_id, model, condition, run_id, effort, mode, experiment_id, input_hash, gold_hash=""):
    """Gold provenance affects scoring identity but is never part of model input."""
    if not experiment_id or type(run_id) is not int or run_id < 0:
        raise ValueError("Explicit experiment ID and nonnegative replicate required")
    return request_hash(dict(case_id=case_id, model=model, condition=condition,
                             run_id=run_id, effort=effort, mode=mode,
                             experiment_id=experiment_id, input_hash=input_hash, gold_hash=gold_hash))


class ResponseContractError(ValueError):
    def __init__(self, reason, metadata):
        super().__init__(reason)
        self.metadata = metadata


def response_metadata(data, requested_model, expected_reasoning=None, expected_tier=None):
    """Preserve billed telemetry before classifying failed/incomplete responses."""
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    values = [usage.get("input_tokens"), usage.get("output_tokens")]
    known = all(type(x) is int and x >= 0 for x in values)
    metadata = {"response_id": data.get("id"), "response_status": data.get("status"),
                "model_version": data.get("model"),
                "input_tokens": values[0] if known else None,
                "output_tokens": values[1] if known else None,
                "usage_status": "known" if known else "unknown", "usage_details": usage,
                "incomplete_details": data.get("incomplete_details"),
                "response_receipt": data, "effective_reasoning":data.get("reasoning"),
                "effective_service_tier":data.get("service_tier")}
    reason = None
    if requested_model == LUNA and data.get("model") != requested_model:
        reason = "Returned model differs from exact requested Luna"
    elif requested_model == LUNA and (expected_reasoning is None or expected_tier != "default"
            or returned_reasoning.get("effort") != expected_reasoning.get("effort")
            or returned_reasoning.get("mode") != expected_reasoning.get("mode")
            or data.get("service_tier") != expected_tier):
        reason = "Effective reasoning effort/mode/tier absent or differs from request"
    elif data.get("status") != "completed":
        reason = "Responses request did not complete"
    elif any(c.get("type") == "refusal" for item in data.get("output", [])
             for c in item.get("content", [])):
        reason = "Model refused request"
    if reason:
        raise ResponseContractError(reason, metadata)
    return metadata


def reconcile_plan(plan, observed):
    """One final outcome per planned episode; absent attempts remain explicit failures."""
    expected = {}
    for row in plan:
        eid = row["episode_id"]
        if eid in expected and expected[eid] != row:
            raise ValueError("Conflicting planned episode")
        expected[eid] = row
    if not expected:
        raise ValueError("No immutable planned episodes")
    found = {}
    for row in observed:
        eid = row["episode_id"]
        if eid not in expected:
            raise ValueError("Observed episode absent from immutable plan")
        for key in ("id", "model", "condition", "run_id", "reasoning_effort", "reasoning_mode",
                    "experiment_id", "input_hash", "configuration_id", "fixture_hash", "gold_hash", "expected", "is_dry_run"):
            value = row.get(key)
            # CSV empty effort is normalized by caller; all other fields must agree.
            if value != expected[eid].get(key):
                raise ValueError("Observed episode identity/gold differs from plan: " + key)
        found[eid] = row
    return [found.get(eid, dict(row, attempt_id="missing:" + eid,
                timestamp=row["planned_at"], predicted="missing", correct=False,
                usage_status="unknown", input_tokens=None, output_tokens=None,
                response_status="not_observed", response_id=None))
            for eid, row in expected.items()]


def selected_ids(available, requested):
    """Stable selection rejects missing or duplicate IDs rather than broadening a pilot."""
    if len(set(available)) != len(available):
        raise ValueError("Duplicate fixture IDs")
    if not requested:
        return list(available)
    if len(set(requested)) != len(requested) or set(requested) - set(available):
        raise ValueError("Unknown or duplicate requested case IDs")
    wanted = set(requested)
    return [x for x in available if x in wanted]
