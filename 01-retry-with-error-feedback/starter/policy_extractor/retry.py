from __future__ import annotations
import logging
from typing import Any
from policy_extractor.client import MessageClient
from policy_extractor.extractor import EXTRACT_POLICY_TOOL, build_extraction_messages, parse_tool_use
from policy_extractor.records import Endorsement, ExtractionOutcome, PolicyExtraction, PremiumComponent, RetryFutileEscalation, ValidationError
from policy_extractor.validator import validate_extraction

logger = logging.getLogger(__name__)

def extract_with_retry(*, client: MessageClient, policy_id: str, document_text: str, max_retries: int = 3, model: str = "claude-haiku-4-5", max_tokens: int = 2048) -> ExtractionOutcome:
    prior_attempts, history, last_err = [], [], None
    for i in range(max_retries + 1):
        msgs, sys = build_extraction_messages(document_text, prior_attempts)
        resp = client.create(model=model, max_tokens=max_tokens, system=sys, messages=msgs, tools=[EXTRACT_POLICY_TOOL], tool_choice={"type": "tool", "name": "extract_policy"})
        ext = parse_tool_use(resp)
        if not (err := validate_extraction(ext)):
            return build_extraction(policy_id=policy_id, extraction=ext, attempt_index=i, history=history)
        history.append(err)
        if err.category == "missing_source":
            return RetryFutileEscalation(
                policy_id=policy_id,
                field=err.field,
                category="missing_source",
                detected_pattern=err.detected_pattern,
                reason=err.message
            )
        prior_attempts.append({"extraction": ext, "error_field": err.field, "error_category": err.category, "error_pattern": err.detected_pattern, "error_message": err.message})
        last_err = err
    return RetryFutileEscalation(
        policy_id=policy_id,
        field=last_err.field,
        category=last_err.category if last_err.category == "missing_source" else "missing_source", # fallback category if needed
        detected_pattern=f"retries_exhausted__{last_err.detected_pattern}",
        reason=f"Retries exhausted after {max_retries} attempts."
    )

def build_extraction(*, policy_id: str, extraction: dict[str, Any], attempt_index: int, history: list[ValidationError]) -> PolicyExtraction:
    e, c = extraction.get("endorsements"), extraction.get("premium_components")
    return PolicyExtraction(
        policy_id=policy_id, policy_type=extraction["policy_type"], premium_amount=extraction["premium_amount"],
        deductible=extraction["deductible"], coverage_limit=extraction["coverage_limit"],
        endorsements=[Endorsement(name=x["name"], limit=x.get("limit")) for x in e] if e is not None else None,
        exclusions=list(extraction.get("exclusions") or []),
        premium_components=[PremiumComponent(name=x["name"], amount=x["amount"]) for x in c] if c is not None else None,
        confidence=dict(extraction.get("confidence") or {}), retry_count=attempt_index, final_attempt_index=attempt_index, validation_history=list(history)
    )
