"""
Execution planning (assignment section 4.2).

Kept deterministic/template-based rather than LLM-generated: the plan's
job is to declare, up front, which tools will run and in what order, so a
reviewer can audit the agent's intended behaviour before it touches data.
An LLM-generated plan could vary run to run for no good reason.
"""
from __future__ import annotations

from agent.schemas import ExecutionPlan, StructuredRequirement


def build_plan(req: StructuredRequirement) -> ExecutionPlan:
    entity_label = {
        "supplier": "suppliers",
        "professional": "professionals",
        "opportunity": "opportunities",
        "business": "businesses",
    }.get(req.entity_type, "entities")

    steps = [
        f"Search {entity_label} by category/skill keywords and requested location",
        f"Inspect candidate {entity_label} for certifications, capacity and availability",
        "Filter out records that fail any hard requirement",
        "Score and rank the remaining candidates using weighted, evidence-backed criteria",
        "Validate every recommendation against the dataset (existence, constraints, duplicates, entity type)",
        "Attempt correction (search again / re-filter) if validation fails, up to 3 times",
        "Prepare the final structured response, including a draft outreach message awaiting approval",
    ]
    return ExecutionPlan(steps=steps)
