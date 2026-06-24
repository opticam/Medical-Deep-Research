# Medical Deep Research Agent

A physician-grade multi-agent literature review system built with **Google ADK** and 
deployed to **Vertex AI Agent Engine**. Adapts the [deep-search](https://github.com/google/adk-samples/tree/main/python/agents/deep-search) architecture for clinical medicine.

## What It Does

A physician provides a clinical research topic (e.g., *"GLP-1 agonists for heart failure in patients with T2DM"*). The agent:

1. **Collaborates** with the physician to refine and approve a research plan
2. **Searches** PubMed/NCBI and FDA openFDA autonomously, section by section
3. **Evaluates** each section's clinical rigor and iteratively refines it
4. **Produces** a structured report with full citations, evidence levels, and clinical recommendations

---

## Architecture

```
interactive_planner_agent (root — Human-in-the-Loop)
│
│  [After plan approval]
│
└── research_pipeline (SequentialAgent)
    │
    ├── 1. section_planner
    │       Converts approved plan → 4-6 section outline
    │
    ├── 2. iterative_refinement_loop (LoopAgent, max 3 iterations)
    │   ├── section_researcher       ← PubMed + FDA search per section
    │   ├── research_evaluator       ← Scores quality (Pydantic structured output)
    │   ├── escalation_checker       ← Custom BaseAgent: terminates loop if score ≥ 7
    │   └── enhanced_search_executor ← Gap-filling search on failed sections
    │
    └── 3. report_composer
            Assembles final report with citations, evidence table, limitations
```

### Key ADK Patterns Used

| Pattern | Where Used |
|---------|-----------|
| `SequentialAgent` | Research pipeline (planner → loop → composer) |
| `LoopAgent` | Per-section iterative refinement |
| Custom `BaseAgent` | `EscalationChecker` — deterministic loop control |
| `BuiltInPlanner` + `ThinkingConfig` | Section researcher and refinement executor |
| Pydantic `output_schema` | `SectionEvaluation` structured output |
| `output_key` | Writing evaluation JSON to session state |
| Human-in-the-Loop (HITL) | Plan approval before research begins |

---

## Data Sources

| Source | API | Auth |
|--------|-----|------|
| PubMed/NCBI | [E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/) | Free; optional API key for higher rate limits |
| FDA Drug Labels | [openFDA /drug/label](https://open.fda.gov/apis/drug/label/) | Free; optional API key |
| FDA Adverse Events | [openFDA /drug/event (FAERS)](https://open.fda.gov/apis/drug/event/) | Free; optional API key |
| FDA Recalls | [openFDA /drug/enforcement](https://open.fda.gov/apis/drug/enforcement/) | Free; optional API key |

---

## Prerequisites

- Python 3.10–3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- Google Cloud project with billing enabled
- APIs enabled: Vertex AI, Cloud Storage
- Authenticated: `gcloud auth application-default login`

---

## Local Development

```bash
# 1. Clone and enter the project
cd medical-deep-research

# 2. Install dependencies
poetry install

# 3. Configure environment
cp .env.example .env
# Edit .env with your GCP project ID and optional API keys

# 4. Run locally with ADK Dev UI
poetry run adk web .

# Open http://localhost:8000 and select "interactive_planner_agent"
```

**Example research topics to test locally:**
- `"Semaglutide for weight loss in patients with heart failure with preserved ejection fraction"`
- `"Metformin safety in chronic kidney disease stage 3b"`
- `"SGLT2 inhibitors vs DPP-4 inhibitors for type 2 diabetes — comparative effectiveness"`

---

## Deploy to Vertex AI Agent Engine

```bash
# 1. Create the staging GCS bucket (one-time)
gsutil mb -p $GOOGLE_CLOUD_PROJECT gs://$GOOGLE_CLOUD_PROJECT-medical-research-staging

# 2. Deploy (takes 3-5 minutes)
poetry run python deployment/deploy.py

# 3. Query the deployed agent interactively
poetry run python deployment/query_deployed.py
```

The deployed agent ID is saved to `.deployed_agent_id` automatically.

---

## Running Tests

```bash
poetry run pytest eval/ -v
```

Tests cover:
- `generate_research_plan` tool logic (section counts, safety goal inclusion)
- PubMed XML parsing (PMID, authors, citation format, edge cases)
- FDA result normalization (drug labels, adverse events)
- `SectionEvaluation` Pydantic schema validation

---

## Extending This Agent

### Add a new data source (e.g., ClinicalTrials.gov)
1. Add a new tool function in `app/tools.py` following the `search_pubmed` pattern
2. Add it to `tools=[...]` in `section_researcher` and `enhanced_search_executor`
3. Update the agent instruction to describe when to call it

### Add a new agent type (e.g., drug interaction checker)
1. Define a new `Agent` in `app/agent.py`  
2. Insert it into `research_pipeline.sub_agents` at the appropriate position
3. Update `report_composer` instruction to include its output

### Switch to a different model
Change `GEMINI_MODEL = "gemini-2.5-pro"` in `app/agent.py`. For cost optimization 
during development, use `"gemini-2.5-flash"`.

---

## Important Disclaimer

This system is an AI-assisted literature synthesis tool intended to support, not 
replace, clinical judgment. All outputs should be reviewed by qualified clinicians 
before informing patient care decisions. The agent does not have access to real-time 
clinical data, patient records, or proprietary medical databases.

---

## Project Structure

```
medical-deep-research/
├── app/
│   ├── __init__.py          # Exports root_agent (required by ADK CLI)
│   ├── agent.py             # All agent definitions + pipeline orchestration
│   └── tools.py             # PubMed, FDA, and plan generator tools
├── deployment/
│   ├── deploy.py            # Vertex AI Agent Engine deployment script
│   └── query_deployed.py    # Interactive CLI for deployed agent
├── eval/
│   └── test_agent.py        # Pytest unit tests for tools and schemas
├── .env.example             # Environment variable template
├── pyproject.toml           # Poetry dependencies
└── README.md
```
