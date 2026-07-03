"""
Deterministic, evidence-backed match scoring.

Weights follow the example scoring method in the assignment (section 6):
  product/skill relevance   30%
  location suitability      20%
  hard-constraint compliance 25%
  availability/capacity     15%
  reputation                10%

Every sub-score is computed from concrete dataset fields, never guessed by
an LLM, so the total is always reproducible and explainable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent.schemas import ScoreBreakdown, StructuredRequirement

WEIGHTS = {
    "product_or_skill_relevance": 0.30,
    "location_suitability": 0.20,
    "hard_constraint_compliance": 0.25,
    "availability_or_capacity": 0.15,
    "reputation": 0.10,
}


def _relevance_score(record: Dict[str, Any], req: StructuredRequirement) -> Tuple[float, List[str]]:
    """Overlap between requested objective/keywords and product_tags / skills."""
    tags = set(t.lower() for t in (record.get("product_tags") or record.get("skills") or []))
    category = str(record.get("category", "")).lower()
    objective_words = set(w.strip(".,") for w in req.objective.lower().split())
    hits = [t for t in tags if any(w in t or t in w for w in objective_words)]
    category_hit = any(w in category or category in w for w in objective_words) if category else False

    score = 0.0
    evidence = []
    if hits:
        score += min(1.0, 0.5 + 0.15 * len(hits))
        evidence.append(f"Matches objective keywords via tags: {sorted(hits)}")
    if category_hit:
        score = max(score, 0.9)
        evidence.append(f"Category '{record.get('category')}' matches objective")
    if not tags and not category:
        evidence.append("No product/skill tags on record to compare against objective")
    return min(score, 1.0), evidence


def _location_score(record: Dict[str, Any], req: StructuredRequirement) -> Tuple[float, List[str]]:
    loc = record.get("location") or {}
    state = str(loc.get("state", "")).lower()
    city = str(loc.get("city", "")).lower()
    wanted = [l.lower() for l in req.hard_constraints.locations] + [l.lower() for l in req.location_requirements]
    if not wanted:
        return 1.0, ["No location constraint specified; treated as neutral match"]
    if state in wanted or city in wanted:
        return 1.0, [f"Located in {loc.get('city')}, {loc.get('state')} which matches requested location(s)"]
    return 0.0, [f"Located in {loc.get('city')}, {loc.get('state')} which does NOT match requested location(s) {req.hard_constraints.locations}"]


def _hard_constraint_score(record: Dict[str, Any], req: StructuredRequirement, constraint_flags: Dict[str, bool]) -> Tuple[float, List[str]]:
    if not constraint_flags:
        return 1.0, ["No hard constraints to check"]
    satisfied = sum(1 for v in constraint_flags.values() if v)
    total = len(constraint_flags)
    evidence = [f"{k}: {'satisfied' if v else 'NOT satisfied'}" for k, v in constraint_flags.items()]
    return (satisfied / total if total else 1.0), evidence


def _availability_score(record: Dict[str, Any], req: StructuredRequirement) -> Tuple[float, List[str]]:
    evidence = []
    score_components = []

    capacity = record.get("max_capacity_units_per_month")
    min_needed = req.hard_constraints.minimum_capacity or req.quantity_or_capacity
    if capacity is not None and min_needed:
        if capacity >= min_needed:
            score_components.append(1.0)
            evidence.append(f"Capacity {capacity}/month covers requested {min_needed} units")
        else:
            score_components.append(0.0)
            evidence.append(f"Capacity {capacity}/month is BELOW requested {min_needed} units")

    delivery = record.get("max_delivery_days")
    max_allowed = req.hard_constraints.maximum_delivery_days or req.deadline_days
    if delivery is not None and max_allowed:
        if delivery <= max_allowed:
            score_components.append(1.0)
            evidence.append(f"Delivery time {delivery} days is within the {max_allowed}-day requirement")
        else:
            score_components.append(0.0)
            evidence.append(f"Delivery time {delivery} days EXCEEDS the {max_allowed}-day requirement")

    availability = record.get("availability")
    if availability is not None:
        if availability == "available":
            score_components.append(1.0)
            evidence.append("Marked as available")
        else:
            score_components.append(0.2)
            evidence.append(f"Availability status: '{availability}'")

    if not score_components:
        return 0.7, ["No capacity/delivery/availability data on record; neutral-leaning score applied"]
    return sum(score_components) / len(score_components), evidence


def _reputation_score(record: Dict[str, Any]) -> Tuple[float, List[str]]:
    rating = record.get("rating")
    reviews = record.get("reviews_count", 0) or 0
    if rating is None:
        return 0.5, ["No rating on file; neutral score applied"]
    # Normalize a 0-5 rating to 0-1, with a small confidence discount for low review counts.
    base = rating / 5.0
    confidence = min(1.0, reviews / 20.0)
    score = base * (0.7 + 0.3 * confidence)
    return min(score, 1.0), [f"Rating {rating}/5 from {reviews} reviews"]


def calculate_match_score(
    record: Dict[str, Any],
    req: StructuredRequirement,
    constraint_flags: Dict[str, bool],
) -> ScoreBreakdown:
    rel, rel_ev = _relevance_score(record, req)
    loc, loc_ev = _location_score(record, req)
    hc, hc_ev = _hard_constraint_score(record, req, constraint_flags)
    avail, avail_ev = _availability_score(record, req)
    rep, rep_ev = _reputation_score(record)

    total = (
        rel * WEIGHTS["product_or_skill_relevance"]
        + loc * WEIGHTS["location_suitability"]
        + hc * WEIGHTS["hard_constraint_compliance"]
        + avail * WEIGHTS["availability_or_capacity"]
        + rep * WEIGHTS["reputation"]
    )

    return ScoreBreakdown(
        product_or_skill_relevance=round(rel, 3),
        location_suitability=round(loc, 3),
        hard_constraint_compliance=round(hc, 3),
        availability_or_capacity=round(avail, 3),
        reputation=round(rep, 3),
        total=round(total, 3),
        evidence=rel_ev + loc_ev + hc_ev + avail_ev + rep_ev,
    )
