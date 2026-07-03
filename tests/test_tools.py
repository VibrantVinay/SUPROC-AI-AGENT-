"""Unit tests for the deterministic tool layer (search, filter, score, validate)."""
import pytest

from agent.data_store import DataStore
from agent.schemas import HardConstraints, StructuredRequirement
from agent.scoring import calculate_match_score
from agent.tools import filter_by_constraints, get_entity_details, search_entities
from agent.validator import validate_recommendations
from agent.schemas import MatchCandidate


@pytest.fixture(scope="module")
def store():
    return DataStore()


def _req(**kwargs) -> StructuredRequirement:
    defaults = dict(
        objective="Find biodegradable food-container suppliers",
        entity_type="supplier",
        hard_constraints=HardConstraints(),
        requested_results=3,
    )
    defaults.update(kwargs)
    return StructuredRequirement(**defaults)


def test_search_entities_finds_known_supplier(store):
    results = search_entities(store, "biodegradable food containers", entity_type="supplier")
    ids = {r["id"] for r in results}
    assert "SUP-001" in ids


def test_get_entity_details_returns_none_for_unknown_id(store):
    assert get_entity_details(store, "SUP-999") is None


def test_get_entity_details_returns_record_for_known_id(store):
    rec = get_entity_details(store, "SUP-005")
    assert rec is not None
    assert rec["name"] == "Chennai Compostables Ltd"


def test_filter_by_constraints_drops_non_matching_location(store):
    req = _req(hard_constraints=HardConstraints(locations=["Karnataka"]))
    all_suppliers = [r for r in store.all() if r["entity_type"] == "supplier"]
    filtered = filter_by_constraints(store, all_suppliers, req)
    states = {item["record"]["location"]["state"] for item in filtered}
    assert states == {"Karnataka"}


def test_filter_by_constraints_enforces_certification(store):
    req = _req(hard_constraints=HardConstraints(certifications=["food-grade"]))
    all_suppliers = [r for r in store.all() if r["entity_type"] == "supplier"]
    filtered = filter_by_constraints(store, all_suppliers, req)
    ids = {item["record"]["id"] for item in filtered}
    assert "SUP-009" not in ids  # no certifications on file
    assert "SUP-014" not in ids  # only ISO-9001, not food-grade


def test_calculate_match_score_is_reproducible(store):
    req = _req(hard_constraints=HardConstraints(locations=["Karnataka"], certifications=["food-grade"]))
    rec = store.get("SUP-001")
    score1 = calculate_match_score(rec, req, {"location": True, "certifications": True})
    score2 = calculate_match_score(rec, req, {"location": True, "certifications": True})
    assert score1.total == score2.total


def test_validate_recommendations_rejects_unknown_entity(store):
    req = _req()
    fake = MatchCandidate(
        entity_id="SUP-999", name="Ghost Supplier", entity_type="supplier",
        score=calculate_match_score(store.get("SUP-001"), req, {}), evidence=[], constraints_checked={},
    )
    result = validate_recommendations([fake], req, store)
    assert not result.passed
    assert any(i.rule == "entity_exists_in_dataset" for i in result.issues)


def test_validate_recommendations_flags_missing_certification_even_if_marketing_claims_it(store):
    req = _req(hard_constraints=HardConstraints(certifications=["food-grade"]))
    rec = store.get("SUP-014")  # notes claim "food safe" but no cert on file
    cand = MatchCandidate(
        entity_id="SUP-014", name=rec["name"], entity_type="supplier",
        score=calculate_match_score(rec, req, {"certifications": False}),
        evidence=[], constraints_checked={"certifications": False},
    )
    result = validate_recommendations([cand], req, store)
    assert not result.passed
    assert any(i.entity_id == "SUP-014" for i in result.issues)


def test_validate_recommendations_detects_duplicate_listing(store):
    req = _req()
    rec10 = store.get("SUP-010")
    rec11 = store.get("SUP-011")  # intentional duplicate of SUP-010
    cand10 = MatchCandidate(entity_id="SUP-010", name=rec10["name"], entity_type="supplier",
                             score=calculate_match_score(rec10, req, {}), evidence=[], constraints_checked={})
    cand11 = MatchCandidate(entity_id="SUP-011", name=rec11["name"], entity_type="supplier",
                             score=calculate_match_score(rec11, req, {}), evidence=[], constraints_checked={})
    result = validate_recommendations([cand10, cand11], req, store)
    assert any(i.rule == "no_duplicate_recommendations" for i in result.issues)
    assert "SUP-011" not in result.valid_candidate_ids


def test_validate_recommendations_flags_prompt_injection_text_without_acting_on_it(store):
    req = _req()
    rec = store.get("SUP-024")  # contains an embedded prompt-injection attempt in notes
    cand = MatchCandidate(entity_id="SUP-024", name=rec["name"], entity_type="supplier",
                           score=calculate_match_score(rec, req, {}), evidence=[], constraints_checked={})
    result = validate_recommendations([cand], req, store)
    assert any(i.rule == "prompt_injection_scan" for i in result.issues)
    # Crucially: the injected text must never cause auto-approval anywhere in this codebase.
    import inspect
    import agent.agent as agent_module
    source = inspect.getsource(agent_module)
    assert "human_approval_required = True" in source
