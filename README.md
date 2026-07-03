# Suproc Local Agentic Search, Matching & Verification Agent

A local AI agent that takes a natural-language business request, searches a
synthetic Suproc-style dataset, ranks and verifies its recommendations, and
prepares next action waiting for human approval.

Built for the Suproc AI Engineering final-round assignment.

## 1. Quick start

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. (Recommended) pull and run a local model
ollama pull qwen3:4b        # or: ollama pull qwen3:1.7b for lower resource use
ollama serve                 # in a separate terminal, if not already running

# 3. Run the agent
python cli.py "We are a sustainable food-packaging startup based in Bengaluru. \
We need three suppliers from South India that can provide food-grade biodegradable \
containers, support an initial order of 10,000 units and deliver within 30 days. \
Explain why each supplier is suitable, identify any missing information and \
prepare an outreach message."
```

If Ollama isn't running, the CLI prints a one-line notice and **automatically
falls back to a deterministic rule-based requirement parser and templated
outreach drafting** — the rest of the pipeline (search, filter, score,
validate, correct) is unaffected, since it never depended on the LLM to
begin with. This means the system, and its test suite, works fully offline.

```bash
python cli.py --json "..."           # machine-readable structured output
python cli.py --results 5 "..."      # override requested result count
echo "..." | python cli.py           # read the request from stdin
python -m pytest tests/ -v           # run the evaluation test suite
```

## 2. Model & system requirements

- Python 3.11+
- Recommended model: `qwen3:4b` via [Ollama](https://ollama.com)
- Low-resource option: `qwen3:1.7b` (set with `--model qwen3:1.7b` or
  `SUPROC_AGENT_MODEL=qwen3:1.7b`)
- No GPU required for `qwen3:1.7b`; `qwen3:4b` runs comfortably on a modern
  laptop CPU, faster with a GPU.
- No external network calls are made by the agent itself — Ollama runs
  locally, and the dataset is local JSON.

## 3. Architecture

```
cli.py                      Command-line entry point (no auto-execution)
agent/
  schemas.py                 Pydantic models: the structured contract every
                              module reads and writes (StructuredRequirement,
                              MatchCandidate, ValidationResult, FinalOutput...)
  data_store.py               Loads dataset/*.json once; the ONLY place that
                              touches raw files. Every entity id referenced
                              anywhere else is checked against this store.
  llm_client.py               Thin Ollama wrapper. Used ONLY for language
                              understanding (parsing) and drafting (outreach
                              text) -- never for facts, scores or validation.
                              Falls back cleanly if Ollama is unreachable.
  requirement_parser.py       Section 4.1: NL request -> StructuredRequirement.
                              Tries the LLM first, cross-checks its output
                              against a deterministic rule-based extractor
                              ("safety net") so a hard constraint stated in
                              plain text can never be silently dropped by
                              the model, and falls back entirely to the
                              rule-based extractor if Ollama is unavailable.
  planner.py                  Section 4.2: builds a short, deterministic
                              execution plan (not LLM-generated, so it is
                              reproducible run to run).
  tools.py                    Section 4.3 required tools: search_entities,
                              get_entity_details, filter_by_constraints, plus
                              optional tools check_availability,
                              get_interaction_history, search_opportunities.
                              Pure, deterministic, dataset-only functions.
  scoring.py                  Optional tool calculate_match_score. Deterministic,
                              weighted, evidence-producing (section 6).
  validator.py                Required tool validate_recommendations (section 7).
                              Deterministic, dataset-grounded checks; also scans
                              free-text fields for prompt-injection attempts
                              WITHOUT ever treating that text as an instruction.
  outreach.py                  Optional tool draft_outreach. Only ever drafts
                              text; there is no send_* function anywhere in
                              this codebase.
  agent.py                    Orchestrator: parse -> plan -> search -> filter
                              -> score -> validate -> correct (<=3 attempts)
                              -> final structured output. Always sets
                              human_approval_required = True.
dataset/
  businesses.json              30 business/supplier records (deliberately
                              includes incomplete, conflicting, duplicate,
                              and prompt-injection-laced records).
  professionals.json           15 professional records.
  opportunities.json           10 opportunity/project/bounty records
                              (including one with missing fields).
tests/
  test_tools.py                Unit tests for search/filter/score/validate.
  test_agent.py                12 end-to-end scenario tests (assignment
                              section 11), run fully offline/deterministically.
traces/
  sample_run_*.txt             Example CLI execution traces (see section 6).
```

### Why the model/tool boundary matters

The LLM is deliberately kept out of anything that decides a fact:

- **Retrieval** (`search_entities`, `get_entity_details`) reads only the
  local JSON dataset — the model can never introduce an entity that isn't
  really there, because nothing downstream trusts model-authored IDs. Every
  recommended `entity_id` is looked up in `DataStore` before it's trusted.
- **Filtering** (`filter_by_constraints`) and **scoring**
  (`calculate_match_score`) are plain, reproducible Python. Given the same
  record and requirement, they always return the same numbers, with a
  human-readable evidence trail for every sub-score.
- **Validation** (`validate_recommendations`) independently re-derives
  whether hard constraints are satisfied and re-sums the score's weighted
  components — it does not just trust what scoring.py claimed.
- The LLM is used only in `requirement_parser.py` (turning English into
  structured JSON) and `outreach.py` (turning structured facts into a
  polite email draft). Both have deterministic fallbacks and both are
  cross-checked / fact-constrained respectively.

## 4. Tool definitions

| Tool | Required? | File | Purpose |
|---|---|---|---|
| `search_entities` | required | `agent/tools.py` | Keyword search over the dataset, optionally scoped by entity type |
| `get_entity_details` | required | `agent/tools.py` | Fetch one record by id, or `None` if it doesn't exist |
| `filter_by_constraints` | required | `agent/tools.py` | Drop records that fail any hard constraint, returning per-record pass/fail flags |
| `validate_recommendations` | required | `agent/validator.py` | Ten deterministic checks (see section 7 below) |
| `calculate_match_score` | optional | `agent/scoring.py` | Weighted, evidence-backed 0–1 score per candidate |
| `check_availability` | optional | `agent/tools.py` | Capacity / delivery-time / availability lookup |
| `get_interaction_history` | optional | `agent/tools.py` | Rating, review count, tenure — reputation signals |
| `search_opportunities` | optional | `agent/tools.py` | Filter the opportunities/projects/bounties subset |
| `draft_outreach` | optional | `agent/outreach.py` | Writes (never sends) an outreach message |

## 5. Ranking / scoring method

Matches the example in assignment section 6:

| Component | Weight | Computed from |
|---|---|---|
| Product/skill relevance | 30% | Overlap between the objective's keywords and the record's `product_tags`/`skills`/`category` |
| Location suitability | 20% | Whether the record's city/state matches a requested location |
| Hard-constraint compliance | 25% | Fraction of hard constraints (`filter_by_constraints` flags) satisfied |
| Availability/capacity | 15% | Capacity vs. requested quantity, delivery days vs. deadline, `availability` field |
| Reputation | 10% | `rating` normalized to 0–1, discounted for low `reviews_count` |

Every sub-score returns a list of `evidence` strings quoting the exact
dataset fields it used, so nothing in the final score is a black box.

## 6. Verification, correction & human approval

`validate_recommendations` runs 10 checks: entity exists in dataset, hard
constraints satisfied, factual claims supported (e.g. a certification claim
in marketing copy without a certificate on file is rejected), no duplicate
recommendations (including near-duplicate listings with the same name and
location, not just identical ids), requested result count met, no
unsupported facts presented as certain, correct entity type, match-score
recomputation sanity check, a prompt-injection scan of every free-text
field, and — implicitly — that the proposed action always requires human
approval.

If validation fails, `agent.py` removes the disqualified candidate(s) and
retries with the next-best candidates from the already-scored pool, up to
**3 correction attempts**. If fewer valid matches exist than requested, the
agent says so explicitly (`validation_status =
"passed_with_fewer_results (...)"` or `"failed_no_valid_matches"`) instead
of inventing a result. See `traces/sample_run_correction_and_dedup.txt` for
a real run where a duplicate listing (`SUP-011`) is caught and excluded.

**Human approval:** `human_approval_required` is unconditionally `True` on
every run. There is no `send_message`, `send_email`, `award_bounty`, or any
other consequential-action function anywhere in this codebase — the agent
is structurally incapable of performing the action it recommends, so this
isn't just a flag that could be bypassed by clever prompting (see the
prompt-injection test in `tests/test_agent.py`).

## 7. Dataset

`dataset/businesses.json` (30 records), `dataset/professionals.json` (15
records), and `dataset/opportunities.json` (10 records) are synthetic and
contain, on purpose:

- **Ambiguous/incomplete records** — e.g. `SUP-009` has no certifications
  on file despite marketing language implying safety; `OPP-010` is missing
  budget, deadline and requirements entirely.
- **Conflicting records** — e.g. `SUP-014` claims "food safe" in its notes
  but only holds an ISO-9001 certificate, not food-grade.
- **Duplicate listings** — `SUP-010` and `SUP-011` are identical listings
  (simulating a data-migration duplicate) to test dedup logic.
- **A prompt-injection attempt** — `SUP-024`'s `notes` field contains text
  instructing the reader to "ignore all previous instructions" and
  auto-approve/send outreach. The validator flags this as a risk but never
  executes it; no code path in this repository can send anything regardless.
- **Out-of-scope records** — e.g. conventional plastic-packaging suppliers
  and a Mumbai-based supplier (outside the "South India" location scope
  used in the example request) to test constraint filtering.

## 8. Evaluation tests

`tests/test_agent.py` and `tests/test_tools.py` implement 23 automated
tests covering the 12 required scenario types from assignment section 11
(normal request, impossible constraints, conflicting requirements, missing
request info, missing dataset info, ambiguous category, duplicates,
invalid/unavailable entity, initial-validation-failure-then-correction,
prompt injection, mandatory human approval, and a request that explicitly
asks the agent to ignore validation).

```bash
python -m pytest tests/ -v
```

**Latest local run: 23 / 23 passed, 0 failed.**

Known limitations of the test suite: the LLM-backed code paths
(`requirement_parser.parse_requirement` and `outreach.draft_outreach` when
a live model is reachable) are exercised only when Ollama is actually
running, since the goal was fully repeatable, model-free CI. If you have
Ollama running locally, re-run the CLI against the sample requests in
`traces/` and compare — the deterministic tool/validation layer behaves
identically either way, only the requirement-parsing quality and prose
wording of the outreach draft improve with the real model.

## 9. Known limitations

- `search_entities` uses substring keyword matching, not semantic/vector
  search sufficient for this dataset's size, but a larger production
  dataset would benefit from embeddings.
- The rule-based requirement-parser fallback recognizes a fixed list of
  South Indian states and certification keywords; it will not generalize
  to, say, North Indian states or certification types outside that list
  without extending the keyword lists in `requirement_parser.py`.
- `filter_by_constraints` treats "minimum capacity" as the supplier's
  monthly capacity; it does not model multi-supplier order-splitting.
- The outreach draft is a single message addressed to all shortlisted
  candidates together rather than one personalized email per supplier;
  this was a scope simplification for the provided window.


