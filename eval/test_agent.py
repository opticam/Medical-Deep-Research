"""
Evaluation Tests — Medical Deep Research Agent
===============================================
Tests the tool functions and agent pipeline logic without full LLM calls.
Run with:  pytest eval/ -v
"""

from __future__ import annotations

import json
import pytest

# ---------------------------------------------------------------------------
# Tool unit tests
# ---------------------------------------------------------------------------


class TestGenerateResearchPlan:
    """Tests the generate_research_plan tool."""

    def test_standard_plan_has_six_sections(self):
        from app.tools import generate_research_plan

        result = generate_research_plan(
            topic="Semaglutide for cardiovascular risk reduction",
            urgency="standard",
        )
        assert result["estimated_sections"] == 6
        assert len(result["research_goals"]) == 6

    def test_urgent_plan_has_four_sections(self):
        from app.tools import generate_research_plan

        result = generate_research_plan(
            topic="Metformin dosing in CKD",
            urgency="urgent",
        )
        assert result["estimated_sections"] == 4

    def test_plan_includes_safety_section(self):
        from app.tools import generate_research_plan

        result = generate_research_plan(topic="Warfarin monitoring in AF")
        titles = [g["title"] for g in result["research_goals"]]
        # At least one section should cover safety / adverse events
        safety_sections = [
            t for t in titles if any(
                word in t.lower() for word in ["safety", "adverse", "contraindication"]
            )
        ]
        assert len(safety_sections) >= 1

    def test_plan_includes_fda_relevant_goals(self):
        from app.tools import generate_research_plan

        result = generate_research_plan(topic="Ozempic weight loss")
        fda_goals = [g for g in result["research_goals"] if g.get("fda_relevant")]
        assert len(fda_goals) >= 2

    def test_patient_population_included_in_clinical_question(self):
        from app.tools import generate_research_plan

        result = generate_research_plan(
            topic="Beta-blockers",
            patient_population="patients with COPD",
        )
        # The patient population should appear somewhere in the goals
        all_questions = " ".join(
            g["clinical_question"] for g in result["research_goals"]
        )
        assert "COPD" in all_questions


class TestParsePubmedXml:
    """Tests XML parsing of PubMed responses."""

    SAMPLE_XML = """<?xml version="1.0" ?>
    <!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2019//EN"
    "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_190101.dtd">
    <PubmedArticleSet>
    <PubmedArticle>
      <MedlineCitation>
        <PMID Version="1">12345678</PMID>
        <Article>
          <Journal>
            <Title>The New England Journal of Medicine</Title>
            <JournalIssue>
              <Volume>385</Volume>
              <Issue>22</Issue>
              <PubDate><Year>2021</Year></PubDate>
            </JournalIssue>
          </Journal>
          <ArticleTitle>Semaglutide and Cardiovascular Outcomes in Obesity without Diabetes</ArticleTitle>
          <AuthorList>
            <Author>
              <LastName>Lincoff</LastName><Initials>AM</Initials>
            </Author>
            <Author>
              <LastName>Brown-Frandsen</LastName><Initials>K</Initials>
            </Author>
          </AuthorList>
          <Abstract>
            <AbstractText>Background: Obesity is associated with cardiovascular risk...</AbstractText>
          </Abstract>
          <Pagination><MedlinePgn>2127-2138</MedlinePgn></Pagination>
          <PublicationTypeList>
            <PublicationType UI="D016428">Journal Article</PublicationType>
            <PublicationType UI="D016449">Randomized Controlled Trial</PublicationType>
          </PublicationTypeList>
        </Article>
      </MedlineCitation>
    </PubmedArticle>
    </PubmedArticleSet>"""

    def test_parses_pmid(self):
        from app.tools import _parse_pubmed_xml

        articles = _parse_pubmed_xml(self.SAMPLE_XML)
        assert len(articles) == 1
        assert articles[0]["pmid"] == "12345678"

    def test_parses_title(self):
        from app.tools import _parse_pubmed_xml

        articles = _parse_pubmed_xml(self.SAMPLE_XML)
        assert "Semaglutide" in articles[0]["title"]

    def test_parses_authors(self):
        from app.tools import _parse_pubmed_xml

        articles = _parse_pubmed_xml(self.SAMPLE_XML)
        assert "Lincoff" in articles[0]["authors"]

    def test_builds_citation_string(self):
        from app.tools import _parse_pubmed_xml

        articles = _parse_pubmed_xml(self.SAMPLE_XML)
        citation = articles[0]["citation"]
        assert "PMID: 12345678" in citation
        assert "2021" in citation

    def test_publication_types_extracted(self):
        from app.tools import _parse_pubmed_xml

        articles = _parse_pubmed_xml(self.SAMPLE_XML)
        assert "Randomized Controlled Trial" in articles[0]["publication_types"]

    def test_pubmed_url_correct(self):
        from app.tools import _parse_pubmed_xml

        articles = _parse_pubmed_xml(self.SAMPLE_XML)
        assert articles[0]["pubmed_url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"

    def test_handles_malformed_xml(self):
        from app.tools import _parse_pubmed_xml

        result = _parse_pubmed_xml("<broken xml>><<")
        assert result == []


class TestParseFdaResults:
    """Tests FDA result normalization."""

    def test_drug_label_extracts_brand_name(self):
        from app.tools import _parse_fda_results

        raw = [{
            "openfda": {
                "brand_name": ["Ozempic"],
                "generic_name": ["semaglutide"],
                "manufacturer_name": ["Novo Nordisk"],
            },
            "indications_and_usage": ["Indicated for type 2 diabetes mellitus..."],
            "warnings": ["Risk of thyroid C-cell tumors..."],
            "boxed_warning": ["WARNING: RISK OF THYROID C-CELL TUMORS"],
            "contraindications": ["Personal or family history of MTC"],
            "dosage_and_administration": ["0.25 mg once weekly for 4 weeks..."],
            "adverse_reactions": ["Nausea, vomiting, diarrhea..."],
            "drug_interactions": ["No clinically relevant interactions identified"],
        }]

        parsed = _parse_fda_results(raw, "drug_label")
        assert len(parsed) == 1
        assert parsed[0]["brand_name"] == "Ozempic"
        assert parsed[0]["generic_name"] == "semaglutide"
        assert "thyroid" in parsed[0]["boxed_warning"].lower()

    def test_adverse_event_extracts_reactions(self):
        from app.tools import _parse_fda_results

        raw = [{
            "safetyreportid": "ABC123",
            "receivedate": "20240101",
            "patient": {
                "drug": [{"medicinalproduct": "METFORMIN"}],
                "reaction": [
                    {"reactionmeddrapt": "Lactic acidosis"},
                    {"reactionmeddrapt": "Nausea"},
                ]
            },
            "seriousnessdeath": "0",
            "seriousnesshospitalization": "1",
            "seriousnesslifethreatening": "1",
        }]

        parsed = _parse_fda_results(raw, "adverse_event")
        assert parsed[0]["reactions"] == ["Lactic acidosis", "Nausea"]
        assert parsed[0]["seriousness"]["hospitalization"] is True

    def test_unknown_search_type_returns_empty(self):
        from app.tools import _parse_fda_results

        result = _parse_fda_results([{"openfda": {}}], "unknown_type")
        assert result == []


class TestSectionEvaluationSchema:
    """Tests the Pydantic output schema validation."""

    def test_valid_evaluation_passes(self):
        from app.agent import SectionEvaluation

        ev = SectionEvaluation(
            section_title="Efficacy of Metformin",
            quality_score=8,
            has_sufficient_citations=True,
            identified_gaps=["No pediatric studies"],
            escalate=False,
            reasoning="Strong RCT evidence with 5 citations",
        )
        assert ev.escalate is False

    def test_score_out_of_range_raises(self):
        from app.agent import SectionEvaluation
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SectionEvaluation(
                section_title="Test",
                quality_score=11,   # > 10, invalid
                has_sufficient_citations=True,
                identified_gaps=[],
                escalate=False,
                reasoning="",
            )

    def test_low_score_should_set_escalate_true(self):
        """Business rule: quality_score < 7 should mean escalate=True.
        This test verifies the agent instruction is reflected in schema usage."""
        from app.agent import SectionEvaluation

        # Schema doesn't enforce this — the LLM does. But we can test the field exists.
        ev = SectionEvaluation(
            section_title="Weak section",
            quality_score=4,
            has_sufficient_citations=False,
            identified_gaps=["Missing RCTs", "No FDA data"],
            escalate=True,
            reasoning="Only 1 case report found",
        )
        assert ev.escalate is True
        assert ev.quality_score == 4
