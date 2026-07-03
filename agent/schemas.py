"""
Pydantic data models used across the agent.

Keeping these in one place gives every tool and the orchestrator a shared,
validated contract for structured data -- this is what "structured inputs
and outputs" (assignment section 14) refers to.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HardConstraints(BaseModel):
    locations: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    minimum_capacity: Optional[int] = None
    maximum_delivery_days: Optional[int] = None
    entity_type: Optional[str] = None  # supplier / professional / opportunity / business


class Preferences(BaseModel):
    sustainable_materials: Optional[bool] = None
    startup_friendly: Optional[bool] = None
    other: List[str] = Field(default_factory=list)


class StructuredRequirement(BaseModel):
    objective: str
    entity_type: str  # supplier | professional | opportunity | business
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    preferences: Preferences = Field(default_factory=Preferences)
    location_requirements: List[str] = Field(default_factory=list)
    budget: Optional[int] = None
    quantity_or_capacity: Optional[int] = None
    deadline_days: Optional[int] = None
    requested_results: int = 3
    raw_request: str = ""
    parse_method: str = "unknown"  # "llm" or "rule_based_fallback"
    ambiguities: List[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    steps: List[str]


class ScoreBreakdown(BaseModel):
    product_or_skill_relevance: float
    location_suitability: float
    hard_constraint_compliance: float
    availability_or_capacity: float
    reputation: float
    total: float
    evidence: List[str] = Field(default_factory=list)


class MatchCandidate(BaseModel):
    entity_id: str
    name: str
    entity_type: str
    score: ScoreBreakdown
    evidence: List[str] = Field(default_factory=list)
    constraints_checked: Dict[str, bool] = Field(default_factory=dict)
    record: Dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    entity_id: Optional[str] = None
    rule: str
    detail: str


class ValidationResult(BaseModel):
    passed: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    valid_candidate_ids: List[str] = Field(default_factory=list)
    checks_run: List[str] = Field(default_factory=list)


class FinalOutput(BaseModel):
    interpreted_requirement: StructuredRequirement
    plan: ExecutionPlan
    recommended_matches: List[MatchCandidate]
    missing_information: List[str]
    risks_or_uncertainties: List[str]
    recommended_next_action: str
    draft_outreach_message: Optional[str]
    validation_status: str
    human_approval_required: bool
    correction_attempts: int
    notes: List[str] = Field(default_factory=list)
