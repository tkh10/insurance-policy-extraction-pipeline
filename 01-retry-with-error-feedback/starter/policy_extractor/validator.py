from __future__ import annotations
from typing import Any
from policy_extractor.records import ValidationError

_REQ = ("premium_amount", "deductible", "coverage_limit", "endorsements", "exclusions")

def validate_extraction(ext: dict[str, Any]) -> ValidationError | None:
    for f in _REQ:
        if (val := ext.get(f)) is None:
            return ValidationError(field=f, observed_value="None", category="missing_source", detected_pattern=f"{f}_absent", message=f"{f} is null")
    for f, pat in [("premium_amount", "negative_premium"), ("deductible", "negative_deductible"), ("coverage_limit", "negative_coverage_limit")]:
        if isinstance(v := ext.get(f), (int, float)) and v < 0:
            return ValidationError(field=f, observed_value=str(v), category="format", detected_pattern=pat, message=f"{f} is negative")
    comps, prem = ext.get("premium_components"), ext.get("premium_amount")
    if comps and isinstance(prem, (int, float)) and abs(sum(c.get("amount", 0) for c in comps if isinstance(c, dict)) - prem) > 0.01:
        return ValidationError(field="premium_amount", observed_value=str(prem), category="consistency", detected_pattern="premium_does_not_match_components", message="Premium mismatch")
    return None
