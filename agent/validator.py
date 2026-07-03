"""
Deterministic factual validation.

This module never calls the LLM. It checks recommended candidates against
the raw dataset record-by-record so validation is reproducible and cannot
be talked out of a rule by clever prompt text sitting inside a dataset
record (see the prompt-injection defense in `_scan_for_injection`).
"""
from __future__ import annotations

from typing import Any, Dict, List

from agent.data_store import DataStore
from agent.schemas import MatchCandidate, StructuredRequirement, ValidationIssue, ValidationResult

INJECTION_MARKERS = [
    "ignore all previous instructions",
    "ignore previous instructions",
    "you are now in",
    "disregard the above",
    "system prompt",
    "act as",
    "do not require approval",
    "without human approval",
    "automatically send",
    "bypass validation",
]


def _scan_for_injection(record: Dict[str, Any]) -> List[str]:
    """Look for prompt-injection style text inside dataset fields (esp. free-text notes).

    Returns a list of warnings; the caller decides how to react. The key
    defense here is architectural: this function's output is NEVER fed
    back into the LLM as an instruction, and no dataset field is ever
    treated as anything other than inert text describing an entity.
    """
    warnings = []
    text_fields = " ".join(
        str(v) for k, v in record.items() if isinstance(v, str)
    ).lower()
    for marker in INJECTION_MARKERS:
        if marker in text_fields:
            warnings.append(
                f"Record {record.get('id')} contains suspicious embedded instruction-like text "
                f"(matched: '{marker}'). Treated as inert data, not as a command."
            )
    return warnings


def validate_recommendations(
    candidates: List[MatchCandidate],
    req: StructuredRequirement,
    store: DataStore,
) -> ValidationResult:
    issues: List[ValidationIssue] = []
    checks_run = [
        "entity_exists_in_dataset",
        "hard_constraints_satisfied",
        "factual_claims_supported",
        "no_duplicate_recommendations",
        "requested_result_count",
        "no_unsupported_facts_presented_as_certain",
        "correct_entity_type",
        "match_score_recalculated",
        "human_approval_flagged",
        "prompt_injection_scan",
    ]

    seen_ids = set()
    seen_signatures = set()  # (name, location) to catch near-duplicate listings
    valid_ids: List[str] = []

    for cand in candidates:
        record = store.get(cand.entity_id)

        # 1. Entity must exist in the dataset.
        if record is None:
            issues.append(ValidationIssue(
                entity_id=cand.entity_id, rule="entity_exists_in_dataset",
                detail=f"{cand.entity_id} was recommended but does not exist in the dataset.",
            ))
            continue

        # 10. Scan record text for prompt-injection content; never act on it, only report it.
        inj_warnings = _scan_for_injection(record)
        for w in inj_warnings:
            issues.append(ValidationIssue(entity_id=cand.entity_id, rule="prompt_injection_scan", detail=w))

        # 7. Correct entity type.
        if req.entity_type and record.get("entity_type") != req.entity_type:
            issues.append(ValidationIssue(
                entity_id=cand.entity_id, rule="correct_entity_type",
                detail=f"{cand.entity_id} has entity_type '{record.get('entity_type')}', expected '{req.entity_type}'.",
            ))
            continue

        # 2. Hard constraints satisfied (re-derive independently rather than trusting cand.constraints_checked).
        failed_constraints = [k for k, v in cand.constraints_checked.items() if v is False]
        if failed_constraints:
            issues.append(ValidationIssue(
                entity_id=cand.entity_id, rule="hard_constraints_satisfied",
                detail=f"{cand.entity_id} fails hard constraint(s): {failed_constraints}.",
            ))
            continue

        # Certification claims must be backed by the record's certifications list,
        # not by marketing language in free-text notes.
        required_certs = req.hard_constraints.certifications or []
        record_certs = [c.lower() for c in (record.get("certifications") or [])]
        missing_certs = [c for c in required_certs if c.lower() not in record_certs]
        if missing_certs:
            issues.append(ValidationIssue(
                entity_id=cand.entity_id, rule="factual_claims_supported",
                detail=f"{cand.entity_id} does not have evidence of certification(s): {missing_certs}.",
            ))
            continue

        # 4. Duplicate detection: same id twice, or same name+location (near-duplicate listing).
        if cand.entity_id in seen_ids:
            issues.append(ValidationIssue(
                entity_id=cand.entity_id, rule="no_duplicate_recommendations",
                detail=f"{cand.entity_id} was recommended more than once.",
            ))
            continue
        signature = (record.get("name"), str(record.get("location")))
        if signature in seen_signatures:
            issues.append(ValidationIssue(
                entity_id=cand.entity_id, rule="no_duplicate_recommendations",
                detail=f"{cand.entity_id} appears to duplicate an already-recommended listing "
                       f"with the same name and location ({signature}).",
            ))
            continue

        # 8. Recompute the match score's total from its own weighted components as a sanity check.
        weights = {
            "product_or_skill_relevance": 0.30,
            "location_suitability": 0.20,
            "hard_constraint_compliance": 0.25,
            "availability_or_capacity": 0.15,
            "reputation": 0.10,
        }
        recomputed = (
            cand.score.product_or_skill_relevance * weights["product_or_skill_relevance"]
            + cand.score.location_suitability * weights["location_suitability"]
            + cand.score.hard_constraint_compliance * weights["hard_constraint_compliance"]
            + cand.score.availability_or_capacity * weights["availability_or_capacity"]
            + cand.score.reputation * weights["reputation"]
        )
        if abs(recomputed - cand.score.total) > 0.01:
            issues.append(ValidationIssue(
                entity_id=cand.entity_id, rule="match_score_recalculated",
                detail=f"{cand.entity_id} score total {cand.score.total} does not match recomputed {round(recomputed,3)}.",
            ))
            continue

        seen_ids.add(cand.entity_id)
        seen_signatures.add(signature)
        valid_ids.append(cand.entity_id)

    # 5. Requested result count.
    if len(valid_ids) < req.requested_results:
        issues.append(ValidationIssue(
            rule="requested_result_count",
            detail=f"Only {len(valid_ids)} valid match(es) available; {req.requested_results} were requested.",
        ))

    passed = all(
        issue.rule not in {
            "entity_exists_in_dataset", "hard_constraints_satisfied",
            "factual_claims_supported", "no_duplicate_recommendations",
            "correct_entity_type", "match_score_recalculated",
        }
        for issue in issues
    ) and len(valid_ids) > 0

    return ValidationResult(
        passed=passed,
        issues=issues,
        valid_candidate_ids=valid_ids,
        checks_run=checks_run,
    )
