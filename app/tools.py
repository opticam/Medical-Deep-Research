"""
Medical Research Tools
=======================
Custom ADK tools for querying:
  - PubMed/NCBI E-utilities API (free, no key required for low volume)
  - FDA openFDA API (free, optional API key for higher rate limits)
  - Internal plan generator

All functions follow ADK tool conventions:
  - Type-annotated parameters and return values
  - Docstrings that become the tool description shown to the LLM
  - Return plain strings or dicts (serializable)
"""

from __future__ import annotations

import os
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
FDA_BASE = "https://api.fda.gov"
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")  # optional, increases rate limits
FDA_API_KEY = os.getenv("FDA_API_KEY", "")    # optional


# ---------------------------------------------------------------------------
# PubMed / NCBI Tool
# ---------------------------------------------------------------------------


def search_pubmed(
    query: str,
    max_results: int = 10,
    publication_types: str = "Clinical Trial,Meta-Analysis,Systematic Review,Review",
    date_range_years: int = 10,
) -> dict[str, Any]:
    """
    Search PubMed/NCBI for peer-reviewed medical literature.

    Performs an E-utilities search and returns structured citation data
    including PMID, title, authors, journal, publication year, and abstract.
    Prioritizes systematic reviews, meta-analyses, and clinical trials.

    Args:
        query: Medical search query (e.g., "metformin type 2 diabetes HbA1c reduction")
        max_results: Maximum number of articles to return (default 10, max 20)
        publication_types: Comma-separated PubMed publication type filters
        date_range_years: Restrict results to papers from the last N years

    Returns:
        Dict with keys:
            - total_found: Total matching records in PubMed
            - articles: List of article dicts with citation details
            - query_used: The exact query sent to PubMed
            - error: Error message string if the request failed (else None)
    """
    max_results = min(max_results, 20)

    # Build date filter
    from datetime import datetime
    current_year = datetime.now().year
    min_year = current_year - date_range_years
    date_filter = f"{min_year}/01/01:{current_year}/12/31[dp]"

    # Build publication type filter
    pub_type_filters = " OR ".join(
        [f'"{pt}"[pt]' for pt in publication_types.split(",")]
    )
    full_query = f"({query}) AND ({pub_type_filters}) AND ({date_filter})"

    params: dict[str, Any] = {
        "db": "pubmed",
        "term": full_query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    try:
        with httpx.Client(timeout=15.0) as client:
            # Step 1: Search for PMIDs
            search_resp = client.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
            search_resp.raise_for_status()
            search_data = search_resp.json()

            esearch = search_data.get("esearchresult", {})
            total_found = int(esearch.get("count", 0))
            pmids = esearch.get("idlist", [])

            if not pmids:
                return {
                    "total_found": 0,
                    "articles": [],
                    "query_used": full_query,
                    "error": None,
                }

            # Step 2: Fetch summaries for those PMIDs
            fetch_params: dict[str, Any] = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }
            if NCBI_API_KEY:
                fetch_params["api_key"] = NCBI_API_KEY

            fetch_resp = client.get(f"{PUBMED_BASE}/efetch.fcgi", params=fetch_params)
            fetch_resp.raise_for_status()

            articles = _parse_pubmed_xml(fetch_resp.text)

        return {
            "total_found": total_found,
            "articles": articles,
            "query_used": full_query,
            "error": None,
        }

    except httpx.HTTPError as e:
        return {
            "total_found": 0,
            "articles": [],
            "query_used": full_query,
            "error": f"PubMed HTTP error: {str(e)}",
        }
    except Exception as e:
        return {
            "total_found": 0,
            "articles": [],
            "query_used": full_query,
            "error": f"Unexpected error: {str(e)}",
        }


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed efetch XML response into a list of article dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return articles

    for article_elem in root.findall(".//PubmedArticle"):
        try:
            medline = article_elem.find("MedlineCitation")
            if medline is None:
                continue

            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"

            article = medline.find("Article")
            if article is None:
                continue

            # Title
            title_elem = article.find("ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"

            # Journal
            journal_elem = article.find("Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown Journal"

            # Year
            year_elem = article.find("Journal/JournalIssue/PubDate/Year")
            if year_elem is None:
                year_elem = article.find("Journal/JournalIssue/PubDate/MedlineDate")
            year = year_elem.text[:4] if year_elem is not None else "N/A"

            # Volume/Issue/Pages
            vol = _text(article, "Journal/JournalIssue/Volume")
            issue = _text(article, "Journal/JournalIssue/Issue")
            pages = _text(article, "Pagination/MedlinePgn")

            # Authors
            author_list = article.find("AuthorList")
            authors = []
            if author_list is not None:
                for author in author_list.findall("Author")[:6]:
                    last = _text(author, "LastName") or ""
                    initials = _text(author, "Initials") or ""
                    if last:
                        authors.append(f"{last} {initials}".strip())
            if len(authors) > 3:
                authors_str = ", ".join(authors[:3]) + " et al."
            else:
                authors_str = ", ".join(authors)

            # Abstract
            abstract_texts = article.findall("Abstract/AbstractText")
            abstract_parts = []
            for ab in abstract_texts:
                label = ab.get("Label", "")
                text = ab.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)[:800]  # truncate for context window

            # Publication types
            pub_types = [
                pt.text
                for pt in article.findall("PublicationTypeList/PublicationType")
                if pt.text
            ]

            articles.append({
                "pmid": pmid,
                "title": title,
                "authors": authors_str,
                "journal": journal,
                "year": year,
                "volume": vol,
                "issue": issue,
                "pages": pages,
                "abstract": abstract,
                "publication_types": pub_types,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "citation": f"{authors_str}. {title}. {journal}. {year}"
                            + (f";{vol}" if vol else "")
                            + (f"({issue})" if issue else "")
                            + (f":{pages}" if pages else "")
                            + f". PMID: {pmid}.",
            })
        except Exception:
            continue  # skip malformed records

    return articles


def _text(elem: ET.Element, path: str) -> str:
    """Helper: find element by path and return its text, or empty string."""
    found = elem.find(path)
    return found.text if found is not None and found.text else ""


# ---------------------------------------------------------------------------
# FDA openFDA Tool
# ---------------------------------------------------------------------------


def search_fda(
    query: str,
    search_type: str = "drug_label",
    limit: int = 5,
) -> dict[str, Any]:
    """
    Search FDA openFDA databases for drug labels, adverse events, or drug recalls.

    Accesses FDA's official open data API to retrieve prescribing information,
    safety communications, boxed warnings, and adverse event reports.

    Args:
        query: Drug name, condition, or safety query (e.g., "metformin lactic acidosis")
        search_type: One of:
            - "drug_label"   — Official prescribing information (indications, warnings, dosing)
            - "adverse_event" — FDA Adverse Event Reporting System (FAERS) data
            - "drug_recall"   — Drug recall notices
        limit: Number of results to return (default 5, max 10)

    Returns:
        Dict with keys:
            - search_type: The type of FDA data searched
            - total_found: Total matching records
            - results: List of result dicts (structure varies by search_type)
            - error: Error message if request failed (else None)
    """
    limit = min(limit, 10)

    endpoint_map = {
        "drug_label": f"{FDA_BASE}/drug/label.json",
        "adverse_event": f"{FDA_BASE}/drug/event.json",
        "drug_recall": f"{FDA_BASE}/drug/enforcement.json",
    }

    if search_type not in endpoint_map:
        return {
            "search_type": search_type,
            "total_found": 0,
            "results": [],
            "error": f"Invalid search_type. Choose from: {list(endpoint_map.keys())}",
        }

    endpoint = endpoint_map[search_type]
    encoded_query = urllib.parse.quote(f'"{query}"')

    params: dict[str, Any] = {
        "search": f"_exists_:{_fda_field(search_type)}+{encoded_query}",
        "limit": limit,
    }
    if FDA_API_KEY:
        params["api_key"] = FDA_API_KEY

    # Simpler fallback query format
    simple_params: dict[str, Any] = {
        "search": query,
        "limit": limit,
    }
    if FDA_API_KEY:
        simple_params["api_key"] = FDA_API_KEY

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(endpoint, params=simple_params)

            if resp.status_code == 404:
                return {
                    "search_type": search_type,
                    "total_found": 0,
                    "results": [],
                    "error": f"No FDA records found for query: {query}",
                }

            resp.raise_for_status()
            data = resp.json()

        meta = data.get("meta", {}).get("results", {})
        total = meta.get("total", 0)
        raw_results = data.get("results", [])

        parsed = _parse_fda_results(raw_results, search_type)

        return {
            "search_type": search_type,
            "total_found": total,
            "results": parsed,
            "error": None,
        }

    except httpx.HTTPStatusError as e:
        return {
            "search_type": search_type,
            "total_found": 0,
            "results": [],
            "error": f"FDA API error {e.response.status_code}: {e.response.text[:200]}",
        }
    except Exception as e:
        return {
            "search_type": search_type,
            "total_found": 0,
            "results": [],
            "error": f"Unexpected error: {str(e)}",
        }


def _fda_field(search_type: str) -> str:
    fields = {
        "drug_label": "openfda.brand_name",
        "adverse_event": "patient.drug.openfda.brand_name",
        "drug_recall": "openfda.brand_name",
    }
    return fields.get(search_type, "openfda.brand_name")


def _parse_fda_results(
    results: list[dict], search_type: str
) -> list[dict[str, Any]]:
    """Normalize FDA API results into a consistent structure."""
    parsed = []

    for r in results:
        openfda = r.get("openfda", {})
        brand_names = openfda.get("brand_name", ["Unknown"])
        generic_names = openfda.get("generic_name", ["Unknown"])

        if search_type == "drug_label":
            parsed.append({
                "brand_name": brand_names[0] if brand_names else "Unknown",
                "generic_name": generic_names[0] if generic_names else "Unknown",
                "manufacturer": openfda.get("manufacturer_name", ["Unknown"])[0]
                                if openfda.get("manufacturer_name") else "Unknown",
                "indications": _first_field(r, "indications_and_usage"),
                "warnings": _first_field(r, "warnings"),
                "boxed_warning": _first_field(r, "boxed_warning"),
                "contraindications": _first_field(r, "contraindications"),
                "dosage": _first_field(r, "dosage_and_administration"),
                "adverse_reactions": _first_field(r, "adverse_reactions"),
                "drug_interactions": _first_field(r, "drug_interactions"),
                "pregnancy_category": _first_field(r, "pregnancy"),
                "source": "FDA Drug Label (openFDA)",
            })

        elif search_type == "adverse_event":
            patient = r.get("patient", {})
            drugs = patient.get("drug", [{}])
            reactions = patient.get("reaction", [])
            parsed.append({
                "report_id": r.get("safetyreportid", "N/A"),
                "receive_date": r.get("receivedate", "N/A"),
                "drugs_involved": [
                    d.get("medicinalproduct", "Unknown") for d in drugs[:3]
                ],
                "reactions": [
                    rxn.get("reactionmeddrapt", "Unknown") for rxn in reactions[:5]
                ],
                "seriousness": {
                    "death": r.get("seriousnessdeath") == "1",
                    "hospitalization": r.get("seriousnesshospitalization") == "1",
                    "life_threatening": r.get("seriousnesslifethreatening") == "1",
                },
                "source": "FDA FAERS (openFDA)",
            })

        elif search_type == "drug_recall":
            parsed.append({
                "brand_name": brand_names[0] if brand_names else "Unknown",
                "generic_name": generic_names[0] if generic_names else "Unknown",
                "recall_reason": r.get("reason_for_recall", "N/A")[:300],
                "classification": r.get("classification", "N/A"),
                "status": r.get("status", "N/A"),
                "recall_initiation_date": r.get("recall_initiation_date", "N/A"),
                "product_description": r.get("product_description", "N/A")[:200],
                "source": "FDA Enforcement Report (openFDA)",
            })

    return parsed


def _first_field(record: dict, key: str, max_len: int = 500) -> str:
    """Extract first element of an FDA list field, truncated."""
    val = record.get(key)
    if isinstance(val, list) and val:
        return val[0][:max_len]
    if isinstance(val, str):
        return val[:max_len]
    return "Not specified"


# ---------------------------------------------------------------------------
# Research Plan Generator Tool
# ---------------------------------------------------------------------------


def generate_research_plan(
    topic: str,
    clinical_context: str = "",
    patient_population: str = "",
    urgency: str = "standard",
) -> dict[str, Any]:
    """
    Generate a structured medical research plan for a given clinical topic.

    Creates a framework of research goals that will guide the subsequent
    multi-agent literature search. The plan is presented to the physician
    for review and modification before research begins.

    Args:
        topic: The clinical topic to research (e.g., "GLP-1 agonists in heart failure")
        clinical_context: Why the physician needs this research (e.g., "patient with 
                         T2DM and recent MI considering semaglutide")
        patient_population: Specific patient demographics or comorbidities to focus on
        urgency: "urgent" (3 sections, broad strokes) or "standard" (5-6 sections, deep)

    Returns:
        Dict with keys:
            - topic: Confirmed research topic
            - clinical_context: Provided context
            - research_goals: List of specific research goal dicts
            - suggested_sources: Recommended databases to prioritize
            - estimated_sections: Number of report sections planned
    """
    # Determine section count based on urgency
    depth = 4 if urgency == "urgent" else 6

    # Build a structured research plan
    base_goals = [
        {
            "goal_id": 1,
            "title": f"Mechanism of Action & Pharmacology of {topic}",
            "clinical_question": f"What is the pharmacological basis for using {topic}?",
            "evidence_priority": ["Review", "Meta-Analysis"],
            "fda_relevant": True,
        },
        {
            "goal_id": 2,
            "title": "Efficacy Evidence from Clinical Trials",
            "clinical_question": f"What do RCTs and meta-analyses show about efficacy of {topic}"
                                 + (f" in {patient_population}" if patient_population else "") + "?",
            "evidence_priority": ["Clinical Trial", "Meta-Analysis", "Systematic Review"],
            "fda_relevant": False,
        },
        {
            "goal_id": 3,
            "title": "Safety Profile, Adverse Events & Contraindications",
            "clinical_question": f"What are the key safety signals, boxed warnings, and "
                                 f"contraindications for {topic}?",
            "evidence_priority": ["Clinical Trial", "Meta-Analysis"],
            "fda_relevant": True,
        },
        {
            "goal_id": 4,
            "title": "Drug Interactions & Special Populations",
            "clinical_question": f"Are there clinically significant drug interactions or "
                                 f"dose adjustments needed for {topic} in special populations "
                                 f"(renal/hepatic impairment, elderly, pregnancy)?",
            "evidence_priority": ["Review", "Clinical Trial"],
            "fda_relevant": True,
        },
        {
            "goal_id": 5,
            "title": "Comparison with Current Standard of Care",
            "clinical_question": f"How does {topic} compare to existing treatments in terms "
                                 f"of efficacy, safety, and cost-effectiveness?",
            "evidence_priority": ["Meta-Analysis", "Systematic Review", "Clinical Trial"],
            "fda_relevant": False,
        },
        {
            "goal_id": 6,
            "title": "Current Guidelines & Evidence Gaps",
            "clinical_question": f"What do current clinical guidelines recommend regarding "
                                 f"{topic}, and what questions remain unanswered?",
            "evidence_priority": ["Guideline", "Systematic Review"],
            "fda_relevant": False,
        },
    ]

    # Add population-specific goal if provided
    if patient_population and urgency != "urgent":
        base_goals.insert(3, {
            "goal_id": 3.5,
            "title": f"Evidence Specific to {patient_population}",
            "clinical_question": f"Is there subgroup evidence for {topic} specifically "
                                 f"in {patient_population}?",
            "evidence_priority": ["Clinical Trial", "Meta-Analysis"],
            "fda_relevant": False,
        })

    research_goals = base_goals[:depth]

    return {
        "topic": topic,
        "clinical_context": clinical_context or "Not specified",
        "patient_population": patient_population or "General adult population",
        "research_goals": research_goals,
        "suggested_sources": ["PubMed/NCBI", "FDA openFDA (drug labels, FAERS)"],
        "estimated_sections": len(research_goals),
        "plan_version": "1.0 — Pending physician approval",
    }
