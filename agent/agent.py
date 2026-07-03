"""
Agent orchestrator.

Wires together requirement understanding -> planning -> search -> filter ->
score -> validate -> correct (<=3 attempts) -> final structured output.

This is where the "agentic" behaviour lives: the LLM only ever assists with
language understanding and drafting (via requirement_parser / outreach); the
control flow, retrieval, filtering, scoring and validation are all plain
deterministic Python so the agent's decisions are reproducible and auditable.

Nothing in this file sends messages, awards bounties, or writes to the
dataset. Every run ends in a *proposed* next action awaiting human approval,
per assignment section 8.
"""
from __future__ import annotations

from typing import List, Optional

from agent.data_store import DataStore, default_store
from agent.llm_client import LLMClient
from agent.outreach import draft_outreach
from agent.planner import build_plan
from agent.requirement_parser import parse_requirement
from agent.schemas import FinalOutput, MatchCandidate, StructuredRequirement
from agent.scoring import calculate_match_score
from agent.tools import filter_by_constraints, search_entities
from agent.validator import validate_recommendations

MAX_CORRECTION_ATTEMPTS = 3


def _search_query_from_requirement(req: StructuredRequirement) -> str:
    parts = [req.objective]
    return " ".join(parts)


def _build_candidates(
    store: DataStore,
    req: StructuredRequirement,
    exclude_ids: Optional[set] = None,
) -> List[MatchCandidate]:
    exclude_ids = exclude_ids or set()
    query = _search_query_from_requirement(req)
    found = search_entities(store, query, entity_type=req.entity_type)
    if not found:
        # Broaden: drop the entity_type filter only if nothing at all was found under it,
        # so we can still report "here is what exists" rather than an empty void.
        found = search_entities(store, query, entity_type=None)

    filtered = filter_by_constraints(store, found, req)

    candidates: List[MatchCandidate] = []
    for item in filtered:
        record = item["record"]
        if record["id"] in exclude_ids:
            continue
        score = calculate_match_score(record, req, item["constraint_flags"])
        candidates.append(MatchCandidate(
            entity_id=record["id"],
            name=record.get("name", record["id"]),
            entity_type=record.get("entity_type", ""),
            score=score,
            evidence=score.evidence,
            constraints_checked=item["constraint_flags"],
            record=record,
        ))
    candidates.sort(key=lambda c: c.score.total, reverse=True)
    return candidates


def _missing_information(req: StructuredRequirement, store: DataStore) -> List[str]:
    missing = []
    if not req.hard_constraints.locations and not req.location_requirements:
        missing.append("No specific location constraint was stated or detected in the request.")
    if not req.hard_constraints.certifications:
        missing.append("No specific certification requirement was stated; results were not filtered by certification.")
    if req.hard_constraints.minimum_capacity is None:
        missing.append("No minimum order/capacity figure was stated or detected.")
    if req.hard_constraints.maximum_delivery_days is None:
        missing.append("No delivery deadline was stated or detected.")
    missing.extend(f"Ambiguity noted during requirement parsing: {a}" for a in req.ambiguities)
    return missing


def _risks_for_candidates(candidates: List[MatchCandidate]) -> List[str]:
    risks = []
    for c in candidates:
        rec = c.record
        if (rec.get("reviews_count") or 0) < 15:
            risks.append(f"{c.entity_id} ({c.name}) has a limited review history ({rec.get('reviews_count', 0)} reviews); confidence in reputation score is lower.")
        if "pending" in str(rec.get("notes", "")).lower() or "unconfirmed" in str(rec.get("notes", "")).lower():
            risks.append(f"{c.entity_id} ({c.name}) has a note flagging unconfirmed or pending information: \"{rec.get('notes')}\"")
    return risks


def run_agent(raw_request: str, requested_results: Optional[int] = None, store: Optional[DataStore] = None,
              llm_client: Optional[LLMClient] = None) -> FinalOutput:
    store = store or default_store
    llm_client = llm_client or LLMClient()

    # 1. Requirement understanding
    req = parse_requirement(raw_request, llm_client=llm_client)
    if requested_results:
        req.requested_results = requested_results

    # 2. Planning
    plan = build_plan(req)

    # 3-5. Search, filter, score (first pass)
    pool = _build_candidates(store, req)

    excluded: set = set()
    attempt = 0
    validation = None
    selected: List[MatchCandidate] = []
    notes: List[str] = []

    while attempt < MAX_CORRECTION_ATTEMPTS:
        attempt += 1
        available_pool = [c for c in pool if c.entity_id not in excluded]
        selected = available_pool[: req.requested_results]

        validation = validate_recommendations(selected, req, store)

        if validation.passed and len(validation.valid_candidate_ids) >= min(req.requested_results, len(available_pool)):
            break

        # Correction: drop candidates that failed a disqualifying rule and try the next-best ones.
        disqualifying_rules = {
            "entity_exists_in_dataset", "hard_constraints_satisfied",
            "factual_claims_supported", "no_duplicate_recommendations",
            "correct_entity_type", "match_score_recalculated",
        }
        newly_excluded = {i.entity_id for i in validation.issues if i.rule in disqualifying_rules and i.entity_id}
        if not newly_excluded:
            # Nothing more we can exclude/retry -- e.g. simply not enough valid candidates exist.
            notes.append(f"Correction attempt {attempt}: no further correctable issues found; "
                          f"{len(validation.valid_candidate_ids)} valid match(es) available.")
            break
        notes.append(f"Correction attempt {attempt}: excluded {sorted(newly_excluded)} due to validation failures and re-ranked remaining candidates.")
        excluded |= newly_excluded

    # Keep only the entities validator judged valid, in score order.
    valid_ids = set(validation.valid_candidate_ids) if validation else set()
    final_candidates = [c for c in selected if c.entity_id in valid_ids]

    missing_info = _missing_information(req, store)
    risks = _risks_for_candidates(final_candidates)
    for issue in (validation.issues if validation else []):
        if issue.rule == "prompt_injection_scan":
            risks.append(issue.detail)

    human_approval_required = True  # always true; the agent never auto-executes consequential actions

    if final_candidates:
        outreach_msg = draft_outreach(final_candidates, req, llm_client=llm_client)
        action = (
            f"Send a procurement enquiry to {', '.join(c.entity_id for c in final_candidates)}."
            if req.entity_type == "supplier" else
            f"Reach out to {', '.join(c.entity_id for c in final_candidates)} regarding: {req.objective}."
        )
        validation_status = (
            "passed" if validation and validation.passed and len(final_candidates) >= req.requested_results
            else f"passed_with_fewer_results ({len(final_candidates)} of {req.requested_results} requested)"
        )
    else:
        outreach_msg = None
        action = ("No valid matches were found that satisfy the hard constraints. "
                  "Recommend broadening the search criteria (e.g. location or certification) or "
                  "posting an open opportunity/bounty instead.")
        validation_status = "failed_no_valid_matches"

    return FinalOutput(
        interpreted_requirement=req,
        plan=plan,
        recommended_matches=final_candidates,
        missing_information=missing_info,
        risks_or_uncertainties=risks,
        recommended_next_action=action,
        draft_outreach_message=outreach_msg,
        validation_status=validation_status,
        human_approval_required=human_approval_required,
        correction_attempts=attempt,
        notes=notes,
    )
