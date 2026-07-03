"""
Requirement understanding (assignment section 4.1).

Converts a free-text business request into a StructuredRequirement.

Tries the local LLM first (it's better at open-ended language understanding),
and falls back to a deterministic rule-based extractor if the model is
unavailable. The fallback also acts as a safety net: it double-checks
that hard constraints mentioned in the raw text were not dropped by the
LLM (see `_merge_llm_with_rule_based_safety_net`).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from agent.llm_client import LLMClient, OllamaUnavailable
from agent.schemas import HardConstraints, Preferences, StructuredRequirement

SOUTH_INDIA_STATES = ["Karnataka", "Tamil Nadu", "Kerala", "Andhra Pradesh", "Telangana", "Puducherry"]

CERTIFICATION_KEYWORDS = [
    "food-grade", "food grade", "fssai", "iso-9001", "iso 9001", "iso-14001",
    "iso 14001", "compostable-astm-d6400", "astm d6400", "organic", "leed",
]

ENTITY_TYPE_KEYWORDS = {
    "supplier": ["supplier", "suppliers", "manufacturer", "vendor"],
    "professional": ["professional", "professionals", "consultant", "freelancer", "expert"],
    "opportunity": ["opportunity", "opportunities", "project", "bounty", "bounties", "gig"],
    "business": ["business", "businesses", "company", "retailer"],
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

SYSTEM_PROMPT = """You are the requirement-understanding module of a business search agent.
Convert the user's natural-language business request into STRICT JSON matching this schema and NOTHING ELSE:
{
  "objective": string,
  "entity_type": "supplier" | "professional" | "opportunity" | "business",
  "hard_constraints": {
    "locations": [string],
    "certifications": [string],
    "minimum_capacity": number or null,
    "maximum_delivery_days": number or null
  },
  "preferences": {
    "sustainable_materials": true | false | null,
    "startup_friendly": true | false | null,
    "other": [string]
  },
  "location_requirements": [string],
  "budget": number or null,
  "quantity_or_capacity": number or null,
  "deadline_days": number or null,
  "requested_results": number,
  "ambiguities": [string]
}
Only include a hard constraint if the user's text actually implies it. Do not invent locations,
certifications or numbers that are not stated or clearly implied. If the request is ambiguous
(e.g. no location given, an unclear entity type, or conflicting requirements), list each
ambiguity in "ambiguities" as a short human-readable string. Respond with JSON only.
"""


def _expand_locations(raw_text: str) -> List[str]:
    text = raw_text.lower()
    found = set()
    if "south india" in text:
        found.update(SOUTH_INDIA_STATES[:4])  # Karnataka, TN, Kerala, AP -- the canonical "South India" 4
    for state in SOUTH_INDIA_STATES:
        if state.lower() in text:
            found.add(state)
    return sorted(found)


def _extract_entity_type(raw_text: str) -> (str, List[str]):
    text = raw_text.lower()
    ambiguities = []
    scores = {}
    for etype, keywords in ENTITY_TYPE_KEYWORDS.items():
        scores[etype] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        ambiguities.append("No explicit entity type (supplier/professional/opportunity/business) found; defaulted to 'supplier'.")
        return "supplier", ambiguities
    tied = [k for k, v in scores.items() if v == scores[best] and v > 0]
    if len(tied) > 1:
        ambiguities.append(f"Multiple possible entity types mentioned ({tied}); using '{tied[0]}'.")
        return tied[0], ambiguities
    return best, ambiguities


def _extract_certifications(raw_text: str) -> List[str]:
    text = raw_text.lower()
    found = []
    for kw in CERTIFICATION_KEYWORDS:
        if kw in text:
            normalized = "food-grade" if "food" in kw else kw.upper().replace(" ", "-")
            if normalized not in found:
                found.append(normalized)
    return found


def _extract_capacity(raw_text: str) -> Optional[int]:
    m = re.search(r"([\d][\d,]{2,})\s*(units|pieces|containers|items)?", raw_text.lower())
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _extract_deadline_days(raw_text: str) -> Optional[int]:
    m = re.search(r"within\s+(\d+)\s*days", raw_text.lower())
    if m:
        return int(m.group(1))
    m = re.search(r"deliver.{0,15}?(\d+)\s*days", raw_text.lower())
    if m:
        return int(m.group(1))
    return None


def _extract_requested_results(raw_text: str) -> int:
    text = raw_text.lower()
    noun_group = r"(suppliers?|professionals?|opportunit(?:y|ies)|businesses?|matches?|candidates?|results?)"
    m = re.search(rf"\b(\d+)\s+{noun_group}\b", text)
    if m:
        return int(m.group(1))
    for word, num in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+{noun_group}\b", text):
            return num
    return 3


def _extract_preferences(raw_text: str) -> Preferences:
    text = raw_text.lower()
    prefs = Preferences()
    if "sustainable" in text or "biodegradable" in text or "eco" in text or "green" in text:
        prefs.sustainable_materials = True
    if "startup" in text:
        prefs.startup_friendly = True
    return prefs


def _extract_objective_sentence(raw_text: str) -> str:
    """Prefer the sentence that states what's being sought ('need', 'looking for', 'want')
    over a generic opening sentence like 'We are a startup based in X.'"""
    sentences = [s.strip() for s in raw_text.replace("\n", " ").split(".") if s.strip()]
    for kw in ("need", "looking for", "want", "require", "searching for"):
        for s in sentences:
            if kw in s.lower():
                return s[:200]
    return sentences[0][:200] if sentences else ""


def rule_based_parse(raw_text: str, requested_entity_type: Optional[str] = None) -> StructuredRequirement:
    """Deterministic fallback extractor. No LLM involved -- safe for offline tests."""
    entity_type, ambiguities = (_extract_entity_type(raw_text) if not requested_entity_type
                                 else (requested_entity_type, []))
    locations = _expand_locations(raw_text)
    if not locations:
        ambiguities.append("No location constraint detected in the request.")
    certifications = _extract_certifications(raw_text)
    capacity = _extract_capacity(raw_text)
    deadline = _extract_deadline_days(raw_text)
    requested_results = _extract_requested_results(raw_text)
    preferences = _extract_preferences(raw_text)

    objective_sentence = _extract_objective_sentence(raw_text)

    return StructuredRequirement(
        objective=objective_sentence or "Understand and fulfil the business request",
        entity_type=entity_type,
        hard_constraints=HardConstraints(
            locations=locations,
            certifications=certifications,
            minimum_capacity=capacity,
            maximum_delivery_days=deadline,
            entity_type=entity_type,
        ),
        preferences=preferences,
        location_requirements=locations,
        budget=None,
        quantity_or_capacity=capacity,
        deadline_days=deadline,
        requested_results=requested_results,
        raw_request=raw_text,
        parse_method="rule_based_fallback",
        ambiguities=ambiguities,
    )


def _dict_to_structured_requirement(data: Dict[str, Any], raw_text: str) -> StructuredRequirement:
    hc = data.get("hard_constraints", {}) or {}
    prefs = data.get("preferences", {}) or {}
    return StructuredRequirement(
        objective=data.get("objective") or "Understand and fulfil the business request",
        entity_type=data.get("entity_type") or "supplier",
        hard_constraints=HardConstraints(
            locations=hc.get("locations") or [],
            certifications=hc.get("certifications") or [],
            minimum_capacity=hc.get("minimum_capacity"),
            maximum_delivery_days=hc.get("maximum_delivery_days"),
            entity_type=data.get("entity_type"),
        ),
        preferences=Preferences(
            sustainable_materials=prefs.get("sustainable_materials"),
            startup_friendly=prefs.get("startup_friendly"),
            other=prefs.get("other") or [],
        ),
        location_requirements=data.get("location_requirements") or hc.get("locations") or [],
        budget=data.get("budget"),
        quantity_or_capacity=data.get("quantity_or_capacity") or hc.get("minimum_capacity"),
        deadline_days=data.get("deadline_days") or hc.get("maximum_delivery_days"),
        requested_results=data.get("requested_results") or 3,
        raw_request=raw_text,
        parse_method="llm",
        ambiguities=data.get("ambiguities") or [],
    )


def _merge_llm_with_rule_based_safety_net(llm_req: StructuredRequirement, raw_text: str) -> StructuredRequirement:
    """Never let the LLM silently drop a hard constraint that's plainly stated in the text.

    If the rule-based extractor finds a location, certification, capacity or
    deadline that the LLM missed, we add it back and note it as a correction.
    """
    fallback = rule_based_parse(raw_text, requested_entity_type=llm_req.entity_type)
    notes = list(llm_req.ambiguities)

    merged_locations = sorted(set(llm_req.hard_constraints.locations) | set(fallback.hard_constraints.locations))
    if set(merged_locations) != set(llm_req.hard_constraints.locations):
        notes.append("Safety net added location(s) present in the raw text but missed by the model.")
    llm_req.hard_constraints.locations = merged_locations
    llm_req.location_requirements = merged_locations or llm_req.location_requirements

    merged_certs = sorted(set(llm_req.hard_constraints.certifications) | set(fallback.hard_constraints.certifications))
    if set(merged_certs) != set(llm_req.hard_constraints.certifications):
        notes.append("Safety net added certification(s) present in the raw text but missed by the model.")
    llm_req.hard_constraints.certifications = merged_certs

    if llm_req.hard_constraints.minimum_capacity is None and fallback.hard_constraints.minimum_capacity:
        llm_req.hard_constraints.minimum_capacity = fallback.hard_constraints.minimum_capacity
        notes.append("Safety net added a minimum-capacity figure present in the raw text but missed by the model.")

    if llm_req.hard_constraints.maximum_delivery_days is None and fallback.hard_constraints.maximum_delivery_days:
        llm_req.hard_constraints.maximum_delivery_days = fallback.hard_constraints.maximum_delivery_days
        notes.append("Safety net added a delivery-deadline figure present in the raw text but missed by the model.")

    llm_req.ambiguities = notes
    return llm_req


def parse_requirement(raw_text: str, llm_client: Optional[LLMClient] = None) -> StructuredRequirement:
    client = llm_client or LLMClient()
    if client.is_available():
        try:
            raw_json = client.generate(raw_text, system=SYSTEM_PROMPT, json_mode=True)
            data = json.loads(raw_json)
            structured = _dict_to_structured_requirement(data, raw_text)
            return _merge_llm_with_rule_based_safety_net(structured, raw_text)
        except (OllamaUnavailable, json.JSONDecodeError, KeyError, TypeError):
            pass
    return rule_based_parse(raw_text)
