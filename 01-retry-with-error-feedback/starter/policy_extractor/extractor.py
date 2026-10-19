"""Extractor orchestration and prompt definitions."""
from __future__ import annotations

from typing import Any
from anthropic.types import ToolUseBlock

SYSTEM_PROMPT = """You are an expert insurance policy data extractor. Extract structured information from the provided policy document into the requested tool call schema accurately. If any field or value is unstated or absent in the document, return null.

Example 1:
Document text: "Premium Amount: $1,250.00"
Extracted: 1250.0

Example 2:
Document text: "Deductible: None specified"
Extracted: null
"""

EXTRACT_POLICY_TOOL = {
    "name": "extract_policy",
    "description": "Extract structured policy details from insurance document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "policy_type": {"type": "string", "enum": ["auto", "home", "umbrella", "other"]},
            "premium_amount": {"type": ["number", "null"]},
            "deductible": {"type": ["number", "null"]},
            "coverage_limit": {"type": ["number", "null"]},
            "endorsements": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "limit": {"type": ["number", "null"]}
                    },
                    "required": ["name"]
                }
            },
            "exclusions": {"type": "array", "items": {"type": "string"}},
            "premium_components": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "amount": {"type": "number"}
                    },
                    "required": ["name", "amount"]
                }
            },
            "confidence": {
                "type": "object",
                "properties": {
                    "policy_type": {"type": "number"},
                    "premium_amount": {"type": "number"},
                    "deductible": {"type": "number"},
                    "coverage_limit": {"type": "number"},
                    "endorsements": {"type": "number"},
                    "exclusions": {"type": "number"}
                },
                "required": ["policy_type", "premium_amount", "deductible", "coverage_limit", "endorsements", "exclusions"]
            }
        },
        "required": ["policy_type", "premium_amount", "deductible", "coverage_limit", "endorsements", "exclusions", "confidence"]
    }
}


def build_extraction_messages(
    document_text: str,
    prior_attempts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Build (messages, system) for a single extraction attempt."""
    user_content = f"<document>\n{document_text.strip()}\n</document>"

    if prior_attempts:
        attempt_blocks = []
        for idx, attempt in enumerate(prior_attempts, 1):
            ef, ec, ep, em, ext_val = attempt['error_field'], attempt['error_category'], attempt['error_pattern'], attempt['error_message'], attempt['extraction']
            block = (
                f'<prior_attempt index="{idx}">\n'
                f"  <extraction>{ext_val}</extraction>\n"
                f'  <validation_error field="{ef}" category="{ec}" detected_pattern="{ep}">\n'
                f"    {em}\n"
                f"  </validation_error>\n"
                f"</prior_attempt>"
            )
            attempt_blocks.append(block)
        
        user_content += "\n\nYour previous attempts were rejected by the validator:\n" + "\n".join(attempt_blocks)

    return [{"role": "user", "content": user_content}], SYSTEM_PROMPT


def parse_tool_use(message: Any) -> dict[str, Any]:
    """Extract the first tool_use block's input dict from a Message."""
    for block in message.content:
        if isinstance(block, ToolUseBlock) or getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    raise ValueError(
        f"No tool_use block found in response (stop_reason={message.stop_reason}, "
        f"blocks={[type(b).__name__ for b in message.content]})"
    )
