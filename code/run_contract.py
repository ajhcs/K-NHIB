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
    if condition in {"websearch", "pdf_websearch", "text_md_websearch"}:
        payload["tools"] = [{"type": "web_search"}]
    return payload


def request_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def episode_id(*, case_id, model, condition, run_id, effort, mode, experiment_id, input_hash):
    if not experiment_id or type(run_id) is not int or run_id < 0:
        raise ValueError("Explicit experiment ID and nonnegative replicate required")
    return request_hash(dict(case_id=case_id, model=model, condition=condition,
                             run_id=run_id, effort=effort, mode=mode,
                             experiment_id=experiment_id, input_hash=input_hash))


def response_metadata(data, requested_model):
    """Incomplete/refused/substituted responses are never ordinary scored answers."""
    if data.get("model") != requested_model and requested_model == LUNA:
        raise ValueError("Returned model differs from exact requested Luna")
    if data.get("status") != "completed":
        raise ValueError("Responses request did not complete")
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise ValueError("Model refused request")
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    fields = [usage.get("input_tokens"), usage.get("output_tokens")]
    known = all(type(x) is int and x >= 0 for x in fields)
    return {"response_id": data.get("id"), "response_status": data.get("status"),
            "model_version": data.get("model"),
            "input_tokens": fields[0] if known else None,
            "output_tokens": fields[1] if known else None,
            "usage_status": "known" if known else "unknown",
            "usage_details": usage,
            "incomplete_details": data.get("incomplete_details")}


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
