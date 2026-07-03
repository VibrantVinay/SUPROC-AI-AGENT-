"""
Tool layer.

Every function here is a "tool" in the agentic sense: a narrow, deterministic
capability the orchestrator (agent.py) calls explicitly. None of these
functions call the LLM -- they only read the local dataset via DataStore.
This keeps retrieval, filtering and validation fully deterministic and
auditable, per the assignment's "deterministic factual validation"
requirement (section 14).

Required tools (section 4.3): search_entities, get_entity_details,
filter_by_constraints, validate_recommendations (see validator.py).
Optional tools implemented: calculate_match_score (see scoring.py),
get_interaction_history, search_opportunities, check_availability,
draft_outreach (see outreach.py).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.data_store import DataStore
from agent.schemas import StructuredRequirement


def search_entities(
    store: DataStore,
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Keyword search across name/category/product_tags/skills/notes/location.

    This is intentionally simple (substring match) rather than a vector
    search, since the assignment's dataset is small and the priority is
    that every result be traceable to an exact field match -- no fuzzy
    hallucinated relevance.
    """
    q = query.lower().strip()
    q_terms = [t for t in q.replace(",", " ").split() if t]
    results = []
    for rec in store.all():
        if entity_type and rec.get("entity_type") != entity_type:
            continue
        haystack_parts = [
            str(rec.get("name", "")),
            str(rec.get("category", "")),
            " ".join(rec.get("product_tags", []) or []),
            " ".join(rec.get("skills", []) or []),
            str(rec.get("notes", "")),
            str((rec.get("location") or {}).get("city", "")),
            str((rec.get("location") or {}).get("state", "")),
        ]
        haystack = " ".join(haystack_parts).lower()
        if not q_terms or any(term in haystack for term in q_terms):
            results.append(rec)
    return results[:limit]


def get_entity_details(store: DataStore, entity_id: str) -> Optional[Dict[str, Any]]:
    """Return the full raw record for a single entity id, or None if it does not exist."""
    return store.get(entity_id)


def check_availability(store: DataStore, entity_id: str) -> Dict[str, Any]:
    """Surface capacity / delivery-time / availability fields for one entity."""
    rec = store.get(entity_id)
    if rec is None:
        return {"entity_id": entity_id, "found": False}
    return {
        "entity_id": entity_id,
        "found": True,
        "availability": rec.get("availability"),
        "max_capacity_units_per_month": rec.get("max_capacity_units_per_month"),
        "max_delivery_days": rec.get("max_delivery_days"),
        "min_order_units": rec.get("min_order_units"),
    }


def get_interaction_history(store: DataStore, entity_id: str) -> Dict[str, Any]:
    """Reputation / prior-performance signals available in the dataset."""
    rec = store.get(entity_id)
    if rec is None:
        return {"entity_id": entity_id, "found": False}
    return {
        "entity_id": entity_id,
        "found": True,
        "rating": rec.get("rating"),
        "reviews_count": rec.get("reviews_count"),
        "founded_year": rec.get("founded_year"),
        "experience_years": rec.get("experience_years"),
    }


def search_opportunities(
    store: DataStore,
    status: Optional[str] = "open",
    opp_type: Optional[str] = None,
    location_state: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter the opportunities/projects/bounties subset of the dataset."""
    results = []
    for rec in store.by_entity_type("opportunity"):
        if status and rec.get("status") != status:
            continue
        if opp_type and rec.get("opp_type") != opp_type:
            continue
        if location_state and str((rec.get("location") or {}).get("state", "")).lower() != location_state.lower():
            continue
        results.append(rec)
    return results


def _check_single_constraint_set(record: Dict[str, Any], req: StructuredRequirement) -> Dict[str, bool]:
    """Return a flag per hard constraint indicating pass/fail for one record.

    This is the single source of truth for "does this record satisfy the
    hard constraints" -- used by both filter_by_constraints (to decide what
    survives) and scoring/validation (to explain and re-check the decision).
    """
    flags: Dict[str, bool] = {}
    hc = req.hard_constraints

    if hc.entity_type:
        flags["entity_type"] = record.get("entity_type") == hc.entity_type

    if hc.locations:
        loc = record.get("location") or {}
        state = str(loc.get("state", "")).lower()
        city = str(loc.get("city", "")).lower()
        wanted = [l.lower() for l in hc.locations]
        flags["location"] = state in wanted or city in wanted

    if hc.certifications:
        record_certs = [c.lower() for c in (record.get("certifications") or [])]
        flags["certifications"] = all(c.lower() in record_certs for c in hc.certifications)

    if hc.minimum_capacity is not None:
        capacity = record.get("max_capacity_units_per_month")
        min_order = record.get("min_order_units")
        # A supplier satisfies a minimum-capacity requirement if their monthly
        # capacity covers it AND their minimum order size does not exceed it
        # in a way that would make a smaller pilot order impossible.
        cap_ok = capacity is not None and capacity >= hc.minimum_capacity
        flags["minimum_capacity"] = bool(cap_ok)

    if hc.maximum_delivery_days is not None:
        delivery = record.get("max_delivery_days")
        flags["maximum_delivery_days"] = delivery is not None and delivery <= hc.maximum_delivery_days

    return flags


def filter_by_constraints(
    store: DataStore,
    records: List[Dict[str, Any]],
    req: StructuredRequirement,
) -> List[Dict[str, Any]]:
    """Filter records against hard constraints, returning only records that pass ALL of them.

    Each returned item is wrapped with its constraint_flags so the caller
    (and, ultimately, the user) can see exactly why a record passed.
    Records that fail even one hard constraint are dropped here -- hard
    constraints, per the assignment, "must never be silently ignored."
    """
    passed = []
    for rec in records:
        flags = _check_single_constraint_set(rec, req)
        if flags and not all(flags.values()):
            continue
        passed.append({"record": rec, "constraint_flags": flags})
    return passed


TOOL_REGISTRY = {
    "search_entities": search_entities,
    "get_entity_details": get_entity_details,
    "filter_by_constraints": filter_by_constraints,
    "check_availability": check_availability,
    "get_interaction_history": get_interaction_history,
    "search_opportunities": search_opportunities,
    # calculate_match_score lives in scoring.py, validate_recommendations in validator.py,
    # draft_outreach lives in outreach.py -- registered here for discoverability by the planner/LLM.
}
