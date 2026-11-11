"""Validate a diagnosis structurally, without judging its scientific claims."""

import math

from .config import exact


def string_list(value, nonempty=False):
    return isinstance(value, list) and (bool(value) or not nonempty) and all(isinstance(v, str) for v in value)


def diagnosis_error(value, records):
    keys = ("conclusion mechanism_hypotheses environment_ids claim_ids evidence_jobs competing_explanations "
            "uncertainty reward_rationale preferred_policy parameter_estimates")
    if not exact(value, keys):
        return "diagnosis requires the exact schema keys"
    if not all(isinstance(value[k], str) for k in ("conclusion", "uncertainty", "reward_rationale")):
        return "diagnosis prose must be strings"
    if not string_list(value["mechanism_hypotheses"], nonempty=True):
        return "mechanism_hypotheses must be a nonempty string list"
    for key, existing in (("environment_ids", records.environments), ("claim_ids", records.claims),
                           ("evidence_jobs", records.jobs)):
        if not string_list(value[key]) or any(v not in existing for v in value[key]):
            return f"{key} must reference existing IDs"
    if value["preferred_policy"] is not None and (
            not isinstance(value["preferred_policy"], str) or value["preferred_policy"] not in records.policies):
        return "preferred_policy must be a registered ID or null"
    competing = value["competing_explanations"]
    if not isinstance(competing, list):
        return "competing_explanations must be a list"
    for item in competing:
        if not exact(item, "explanation test_jobs status reason") or not all(
                isinstance(item[k], str) for k in ("explanation", "reason", "status")):
            return "invalid competing explanation"
        if item["status"] not in ("supported", "rejected", "untested") or not string_list(item["test_jobs"]) or any(
                j not in records.jobs for j in item["test_jobs"]):
            return "invalid competing explanation status or test jobs"
    estimates = value["parameter_estimates"]
    if not isinstance(estimates, dict):
        return "parameter_estimates must be an object"
    for name, estimate in estimates.items():
        if (not isinstance(name, str) or not exact(estimate, "value unit")
                or type(estimate["value"]) not in (int, float) or not math.isfinite(estimate["value"])
                or not isinstance(estimate["unit"], str)):
            return "parameter estimate requires a finite number and string unit"
    return None
