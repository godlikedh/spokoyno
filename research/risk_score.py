"""Shared display policy; a ranking score is not a calibrated probability."""

import math

POLICY = {
    "maybe": 0.6,
    "alert": 0.8,
    "meaning": "uncalibrated evidence, not probability",
}


def risk_tier(score: float | None) -> str:
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        return "unknown"
    return (
        "alert"
        if score >= POLICY["alert"]
        else "maybe"
        if score >= POLICY["maybe"]
        else "low"
    )
