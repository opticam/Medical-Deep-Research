"""
Medical Deep Research Agent
============================
A multi-agent system for physician-grade deep research using the Google ADK.
Mirrors the deep-search architecture with a 5-stage sequential pipeline:

  1. interactive_planner_agent   — HITL: collaborates with doctor to define research goals
  2. section_planner             — Converts approved plan → structured report outline
  3. section_researcher          — Searches PubMed + FDA per section (LoopAgent inside)
  4. research_evaluator          — Scores section quality; triggers refinement if needed
  5. report_composer             — Assembles final structured report with citations

Deployment target: Vertex AI Agent Engine
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from google.adk.agents import Agent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models.lite_llm import LiteLlm
from google.adk.planners import BuiltInPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from app.tools import (
    generate_research_plan,
    search_fda,
    search_pubmed,
)

# ---------------------------------------------------------------------------
# Model config — Gemini 2.5 Pro via Vertex AI
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-pro"

# ---------------------------------------------------------------------------
# Pydantic schemas for structured outputs
# ---------------------------------------------------------------------------


class SectionEvaluation(BaseModel):
    """Structured output from the research_evaluator agent."""

    section_title: str = Field(description="The section being evaluated")
    quality_score: int = Field(
        description="Quality score 1-10 (clinical rigor, source quality, completeness)",
        ge=1,
        le=10,
    )
    has_sufficient_citations: bool = Field(
        description="True if the section has at least 3 peer-reviewed citations"
    )
    identified_gaps: list[str] = Field(
        description="List of specific gaps or missing clinical evidence"
    )
    escalate: bool = Field(
        description="True if quality_score < 7 or has_sufficient_citations is False"
    )
    reasoning: str = Field(description="Brief clinical rationale for the score")


# ---------------------------------------------------------------------------
# Custom BaseAgent: EscalationChecker
# Controls LoopAgent termination without calling the LLM
# ---------------------------------------------------------------------------


class EscalationChecker(BaseAgent):
    """
    Reads the most recent SectionEvaluation from session state.
    Signals loop escalation (termination) when quality is sufficient.
    No LLM call — pure deterministic control flow.
    """

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        evaluation_json = ctx.session.state.get("last_section_evaluation", "{}")
        try:
            evaluation = json.loads(evaluation_json)
            should_escalate = not evaluation.get("escalate", True)
        except (json.JSONDecodeError, AttributeError):
            should_escalate = False

        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_escalate),
        )


# ---------------------------------------------------------------------------
# Stage 1 — Interactive Planner Agent (Human-in-the-Loop)
# ---------------------------------------------------------------------------

interactive_planner_agent = Agent(
    name="interactive_planner_agent",
    model=GEMINI_MODEL,
    description=(
        "Root orchestrator. Collaborates with the physician to produce an approved "
        "medical research plan, then delegates to the research pipeline."
    ),
    instruction="""You are a senior clinical research coordinator helping a physician 
design a rigorous medical literature review.

Your workflow has three phases:

**PHASE 1 — Draft Plan**
When the physician provides a research topic:
1. Call the `generate_research_plan` tool to create an initial plan.
2. Present the plan clearly, explaining each research goal.
3. Ask for feedback: "Does this plan capture what you need? Would you like to add, 
   remove, or modify any goals?"

**PHASE 2 — Refine**
Incorporate the physician's feedback. Continue refining until they explicitly approve.
Use phrases like "Shall I proceed with this plan?" to seek confirmation.

**PHASE 3 — Delegate**
Once approved, say: "Plan approved. Beginning deep research now..." then 
transfer to the `research_pipeline` sub-agent with the approved plan.

**Guidelines:**
- Always ask about specific patient population (age, comorbidities) if not specified
- Suggest including both efficacy AND safety/adverse event sections
- Recommend a section on contraindications and drug interactions when FDA data is relevant
- Frame all goals as: what the clinician needs to decide or understand
""",
    tools=[generate_research_plan],
    sub_agents=["research_pipeline"],  # resolved below via forward reference
)

# ---------------------------------------------------------------------------
# Stage 2 — Section Planner
# ---------------------------------------------------------------------------

section_planner = Agent(
    name="section_planner",
    model=GEMINI_MODEL,
    description="Converts an approved research plan into a structured report outline.",
    instruction="""You are a medical editor. Your job is to transform an approved 
physician research plan into a clean, structured report outline.

**Input:** The approved research plan from conversation history.

**Output format (markdown):**
```
# [Report Title]

## Outline

### 1. [Section Title]
**Clinical Question:** [Specific question this section answers]
**Key Concepts:** [comma-separated search terms for PubMed/FDA]
**Evidence Type:** [RCTs | Meta-analyses | Case series | FDA labels | Guidelines]

### 2. [Section Title]
...
```

**Rules:**
- Create 4-6 sections. No more, no less.
- Every section must have a clear clinical question.
- Include one dedicated section for "Safety, Adverse Events & Contraindications"
- Include one dedicated section for "Clinical Recommendations & Evidence Gaps"
- Do NOT do any research yourself — only structure the outline.

Save the outline to session state key `report_outline`.
""",
    include_contents="default",
)

# ---------------------------------------------------------------------------
# Stage 3a — Section Researcher (called per section inside a loop)
# ---------------------------------------------------------------------------

section_researcher = Agent(
    name="section_researcher",
    model=GEMINI_MODEL,
    description=(
        "Searches PubMed and FDA databases for a single report section "
        "and writes up the findings with full citations."
    ),
    planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(include_thoughts=True)),
    instruction="""You are a clinical research analyst conducting a literature review 
for a specific report section.

**Your task:**
1. Read the current section from session state (`current_section`).
2. Use `search_pubmed` to find peer-reviewed evidence (call it 2-3 times with 
   different query angles: e.g., mechanism, efficacy, population subgroups).
3. Use `search_fda` to find relevant drug labeling, safety communications, 
   or approval data.
4. Synthesize findings into a well-structured section draft.

**Output format for each section:**
```markdown
### [Section Title]

**Clinical Summary**
[2-3 paragraph narrative synthesizing the evidence]

**Key Findings**
- [Finding 1 with citation]
- [Finding 2 with citation]
- [Finding 3 with citation]

**PubMed Citations**
1. [Authors]. [Title]. [Journal]. [Year];[Vol]([Issue]):[Pages]. PMID: [ID].
2. ...

**FDA References**
- [Drug/Device Name]: [Brief note on labeling/safety communication, with link]
```

**Quality standards:**
- Minimum 3 PubMed citations per section
- Prefer systematic reviews and RCTs over case reports
- Always note the evidence level (e.g., "Level I: RCT", "Level III: Expert opinion")
- Flag any contradictory findings explicitly

Save your section draft to session state key `current_section_draft`.
""",
    tools=[search_pubmed, search_fda],
)

# ---------------------------------------------------------------------------
# Stage 3b — Research Evaluator (structured output)
# ---------------------------------------------------------------------------

research_evaluator = Agent(
    name="research_evaluator",
    model=GEMINI_MODEL,
    description="Evaluates the quality of a researched section using clinical standards.",
    output_schema=SectionEvaluation,
    output_key="last_section_evaluation",
    instruction="""You are a senior physician peer-reviewer evaluating a section draft 
for clinical accuracy and completeness.

Read `current_section_draft` from session state and evaluate it strictly.

**Scoring rubric (1-10):**
- 9-10: Comprehensive, multiple high-quality RCTs/meta-analyses, no gaps
- 7-8:  Good coverage, mostly peer-reviewed, minor gaps noted
- 5-6:  Some gaps, over-reliant on case reports or expert opinion
- 1-4:  Major gaps, insufficient citations, or clinically misleading

**Set escalate=True if:**
- quality_score < 7, OR
- has_sufficient_citations is False (fewer than 3 peer-reviewed sources)

Be strict. Physicians will use this report for clinical decision-making.
""",
    include_contents="none",
)

# ---------------------------------------------------------------------------
# Stage 3c — Enhanced Search Executor (refinement pass)
# ---------------------------------------------------------------------------

enhanced_search_executor = Agent(
    name="enhanced_search_executor",
    model=GEMINI_MODEL,
    description="Performs a targeted refinement search to fill gaps identified by the evaluator.",
    planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(include_thoughts=True)),
    instruction="""You are conducting a targeted gap-filling search for a medical 
literature section.

Read from session state:
- `current_section_draft` — the existing draft
- `last_section_evaluation` — the evaluator's JSON with identified_gaps

**Your task:**
1. For each gap in `identified_gaps`, perform additional `search_pubmed` and/or 
   `search_fda` calls targeting that specific gap.
2. Integrate the new findings into the existing draft.
3. Ensure citations are added for any new evidence.
4. Update `current_section_draft` in session state with the improved version.

Focus only on filling the identified gaps — do not rewrite content that was 
already sufficient.
""",
    tools=[search_pubmed, search_fda],
)

# ---------------------------------------------------------------------------
# Stage 3 — Iterative Refinement Loop (per section)
# ---------------------------------------------------------------------------

iterative_refinement_loop = LoopAgent(
    name="iterative_refinement_loop",
    description="Loops: research → evaluate → refine until quality threshold met (max 3 iterations).",
    max_iterations=3,
    sub_agents=[
        section_researcher,
        research_evaluator,
        EscalationChecker(name="escalation_checker"),
        enhanced_search_executor,
    ],
)

# ---------------------------------------------------------------------------
# Stage 4 — Report Composer
# ---------------------------------------------------------------------------

report_composer = Agent(
    name="report_composer",
    model=GEMINI_MODEL,
    description="Assembles all researched sections into a final physician-grade report.",
    instruction="""You are a chief medical editor assembling a final literature review 
report for a physician.

Read from session state:
- `report_outline` — the approved structure
- `completed_sections` — list of all finalized section drafts

**Assemble the final report with this structure:**

```markdown
# Medical Research Report: [Topic]
**Prepared for:** Physician Review  
**Date:** [Today's date]  
**Research Scope:** [Summary of what was investigated]  
**Evidence Base:** PubMed/NCBI peer-reviewed literature + FDA databases

---

## Executive Summary
[3-5 sentence clinical synopsis: key findings, strength of evidence, 
 primary recommendation]

---

## Table of Contents
[Auto-generated from sections]

---

[All sections in order, formatted consistently]

---

## Overall Evidence Assessment
| Section | Citations | Evidence Level | Quality Score |
|---------|-----------|----------------|---------------|
| ...     | ...       | ...            | ...           |

## Limitations & Gaps in Current Evidence
[Honest assessment of what the literature does NOT yet answer]

## Suggested Next Steps for the Clinician
[3-5 actionable suggestions: additional tests, specialist referrals, 
 monitoring parameters, patient counseling points]

---

*Disclaimer: This report is an AI-assisted literature synthesis intended to 
support, not replace, clinical judgment. All treatment decisions should be 
made in consultation with appropriate specialists and current institutional 
protocols.*
```

Save the complete report to session state key `final_report`.
Output the full report as your response.
""",
    include_contents="none",
)

# ---------------------------------------------------------------------------
# Research Pipeline — Sequential orchestration
# ---------------------------------------------------------------------------

research_pipeline = SequentialAgent(
    name="research_pipeline",
    description=(
        "Executes the full medical research pipeline: outline → research each "
        "section with iterative refinement → compose final report."
    ),
    sub_agents=[
        section_planner,
        iterative_refinement_loop,
        report_composer,
    ],
)

# ---------------------------------------------------------------------------
# Wire sub_agents forward reference on the root agent
# ---------------------------------------------------------------------------

interactive_planner_agent.sub_agents = [research_pipeline]

# ---------------------------------------------------------------------------
# ADK App export (required for Vertex AI Agent Engine)
# ---------------------------------------------------------------------------

root_agent = interactive_planner_agent
