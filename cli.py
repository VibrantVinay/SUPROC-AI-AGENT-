#!/usr/bin/env python3
"""
Command-line interface for the Suproc local agentic search/matching/verification system.

Usage:
    python cli.py "We are a sustainable food-packaging startup based in Bengaluru. ..."
    python cli.py --results 3 "..."
    echo "..." | python cli.py

The agent NEVER performs the recommended next action automatically. It only
prints a structured recommendation and waits for you to approve it manually.
"""
from __future__ import annotations

import argparse
import json
import sys

from agent.agent import run_agent
from agent.llm_client import LLMClient


def _print_section(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    parser = argparse.ArgumentParser(description="Suproc local agentic search & verification agent")
    parser.add_argument("request", nargs="?", help="Natural-language business request")
    parser.add_argument("--results", type=int, default=None, help="Override requested number of results")
    parser.add_argument("--json", action="store_true", help="Print the full structured output as JSON")
    parser.add_argument("--model", default=None, help="Ollama model name to use (default: qwen3:4b)")
    args = parser.parse_args()

    raw_request = args.request
    if not raw_request:
        if not sys.stdin.isatty():
            raw_request = sys.stdin.read().strip()
        if not raw_request:
            parser.error("Provide a request as an argument or via stdin.")

    llm_client = LLMClient(model=args.model) if args.model else LLMClient()
    if not llm_client.is_available():
        print("[notice] Local Ollama model not reachable -- using deterministic rule-based "
              "requirement parsing and templated drafting instead. Run `ollama serve` with the "
              "model pulled (see README) for full LLM-assisted understanding.", file=sys.stderr)

    result = run_agent(raw_request, requested_results=args.results, llm_client=llm_client)

    if args.json:
        print(result.model_dump_json(indent=2))
        return

    req = result.interpreted_requirement
    _print_section("1. Interpreted Business Requirement")
    print(f"Objective       : {req.objective}")
    print(f"Entity type     : {req.entity_type}")
    print(f"Parsed by       : {req.parse_method}")
    print(f"Requested count : {req.requested_results}")

    _print_section("2. Hard Constraints")
    print(json.dumps(req.hard_constraints.model_dump(), indent=2))

    _print_section("3. Preferences")
    print(json.dumps(req.preferences.model_dump(), indent=2))

    _print_section("4. Execution Plan")
    for i, step in enumerate(result.plan.steps, 1):
        print(f"  {i}. {step}")

    _print_section("5. Recommended Matches")
    if not result.recommended_matches:
        print("  No valid matches found. See 'Recommended Next Action' below.")
    for c in result.recommended_matches:
        print(f"\n  [{c.entity_id}] {c.name}  (score: {c.score.total})")
        print(f"    Score breakdown: relevance={c.score.product_or_skill_relevance} "
              f"location={c.score.location_suitability} constraints={c.score.hard_constraint_compliance} "
              f"availability={c.score.availability_or_capacity} reputation={c.score.reputation}")
        print("    Evidence:")
        for e in c.evidence:
            print(f"      - {e}")

    _print_section("6. Missing Information")
    for m in result.missing_information:
        print(f"  - {m}")
    if not result.missing_information:
        print("  None noted.")

    _print_section("7. Risks / Uncertainties")
    for r in result.risks_or_uncertainties:
        print(f"  - {r}")
    if not result.risks_or_uncertainties:
        print("  None noted.")

    _print_section("8. Validation Status")
    print(f"  {result.validation_status}  (correction attempts used: {result.correction_attempts})")
    if result.notes:
        for n in result.notes:
            print(f"  note: {n}")

    _print_section("9. Recommended Next Action")
    print(f"  {result.recommended_next_action}")
    print(f"  Human approval required: {result.human_approval_required}")
    print("  STATUS: Awaiting user approval. No message has been sent and no record has been changed.")

    if result.draft_outreach_message:
        _print_section("10. Draft Outreach Message (NOT sent)")
        print(result.draft_outreach_message)

    print()


if __name__ == "__main__":
    main()
