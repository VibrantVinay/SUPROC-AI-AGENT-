"""
draft_outreach tool (assignment section 4.3, optional tool list).

Always DRAFTS text only. This module has no network/email-sending
capability at all -- there is nothing to accidentally wire up to "send".
Human approval is enforced structurally: the CLI/agent only ever calls
`draft_outreach`, never anything named `send_*`.
"""
from __future__ import annotations

from typing import List

from agent.llm_client import LLMClient, OllamaUnavailable
from agent.schemas import MatchCandidate, StructuredRequirement

SYSTEM_PROMPT = """You write short, professional B2B outreach messages on behalf of a business
that is buying from suppliers/professionals found on the Suproc platform. Only mention facts
given to you about the recipients. Do not invent claims. Keep it under 150 words. Output plain
text only, no markdown."""


def _template_draft(candidates: List[MatchCandidate], req: StructuredRequirement) -> str:
    names = ", ".join(f"{c.name} ({c.entity_id})" for c in candidates) or "the shortlisted candidates"
    location_line = f" in {', '.join(req.hard_constraints.locations)}" if req.hard_constraints.locations else ""
    cert_line = f" with {', '.join(req.hard_constraints.certifications)} certification" if req.hard_constraints.certifications else ""
    qty_line = f" for an initial order of {req.hard_constraints.minimum_capacity} units" if req.hard_constraints.minimum_capacity else ""
    deadline_line = f", deliverable within {req.hard_constraints.maximum_delivery_days} days" if req.hard_constraints.maximum_delivery_days else ""

    return (
        f"Subject: Procurement enquiry -- {req.objective}\n\n"
        f"Hello,\n\n"
        f"We are reaching out{location_line} regarding {req.objective.lower()}{cert_line}. "
        f"We are evaluating suppliers such as {names} and would like to confirm availability"
        f"{qty_line}{deadline_line}.\n\n"
        f"Could you share current pricing, lead times and any certification documents on file? "
        f"We'd appreciate a response so we can finalize our shortlist.\n\n"
        f"Best regards,\n"
        f"[Your name]\n"
        f"[Your company]"
    )


def draft_outreach(
    candidates: List[MatchCandidate],
    req: StructuredRequirement,
    llm_client: LLMClient = None,
) -> str:
    """Return a draft outreach message string. Never sends it anywhere."""
    if not candidates:
        return ""

    client = llm_client or LLMClient()
    if client.is_available():
        try:
            facts = "\n".join(
                f"- {c.name} ({c.entity_id}): {', '.join(c.evidence[:3])}" for c in candidates
            )
            prompt = (
                f"Business objective: {req.objective}\n"
                f"Requested from: {', '.join(req.hard_constraints.locations) or 'no specific location'}\n"
                f"Candidates to reach out to:\n{facts}\n\n"
                f"Write the outreach message now."
            )
            return client.generate(prompt, system=SYSTEM_PROMPT).strip()
        except OllamaUnavailable:
            pass
    return _template_draft(candidates, req)
