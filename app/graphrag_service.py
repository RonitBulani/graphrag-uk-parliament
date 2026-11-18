
"""GraphRAG API service: single-pass engine + topic-aware temporal retrieval.

Paste into app/graphrag_service.py. Existing database is read-only.
This version preserves the existing API response fields and citation ordering.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from app.evidence_checks import check_evidence_coverage
from app.general_grounded_generation import generate_general_grounded_answer
from app.question_aware_retrieval import retrieve_question_aware_evidence
from app.temporal_grounded_v2 import generate_grounded_answer
from app.temporal_retrieval import build_year_date_index, retrieve_year_candidates
from src.graphrag_engine_optimized import GraphRAGEngine

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DATA_START_YEAR = 2015
DATA_END_YEAR = 2019

YEAR_RANGE = re.compile(
    r"\b(?:between|from)\s+(20\d{2})\s+(?:and|to|through|-)\s+(20\d{2})\b",
    re.IGNORECASE,
)

SCOTTISH_TOPIC = re.compile(
    r"indyref2|scottish independence|independence referendum|"
    r"claim of right for scotland",
    re.IGNORECASE,
)

CUSTOMS_CUE = re.compile(
    r"leave|leaving|remain|remaining|stay|permanent|comprehensive|"
    r"independent trade|jobs|tariff|border|frictionless|rules of origin|"
    r"trade deal|trade policy|agreement|single market|alignment|"
    r"indicative vote|motion|amendment|membership|withdrawal|"
    r"retain|support|oppose|outside|trade barrier",
    re.IGNORECASE,
)

VOTE_TOPIC = re.compile(
    r"second (?:brexit |eu )?referendum|another referendum|"
    r"people['’]s vote|public vote|confirmatory ballot|"
    r"final referendum|decision referendum|referendum on (?:the |a )?deal",
    re.IGNORECASE,
)

VOTE_CUE = re.compile(
    r"support|oppose|reject|against|should|must|would|not have|"
    r"hold|offer|proposal|amendment|confirmatory|deal|"
    r"democra|mandate|vote|electorate|public|negotiat|refuse",
    re.IGNORECASE,
)

# Different search formulations for each requested year.
TEMPORAL_QUERIES = {
    "customs": (
        "UK leave customs union independent international trade policy Brexit",
        "Labour permanent comprehensive customs union jobs frictionless EU trade",
        "customs union amendments indicative votes 2019 trade policy Northern Ireland",
    ),
    "referendum": (
        "second Brexit referendum support public vote confirmatory ballot on EU deal",
        "oppose second Brexit referendum uphold 2016 vote democracy mandate",
        "People's Vote Labour Liberal Democrat Conservative parliamentary amendment",
    ),
}


# ============================================================
# DATE VALIDATION
# ============================================================

def check_year_coverage(question: str) -> int | None:
    """Return an out-of-dataset year if one is explicitly requested."""

    match = YEAR_RANGE.search(question)

    if match:
        start, end = map(int, match.groups())

        if start < DATA_START_YEAR or start > DATA_END_YEAR:
            return start

        if end < DATA_START_YEAR or end > DATA_END_YEAR:
            return end

        return None

    one = re.search(
        r"\b(?:in|during)\s+(20\d{2})\b",
        question,
        re.I,
    )

    if one:
        year = int(one.group(1))

        if not DATA_START_YEAR <= year <= DATA_END_YEAR:
            return year

    return None


# ============================================================
# ENGINE INITIALISATION
# ============================================================

@lru_cache(maxsize=1)
def get_engine() -> GraphRAGEngine:
    return GraphRAGEngine(
        ROOT / "data" / "knowledge_graph.graphml",
        ROOT / "data" / "knowledge_graph_metadata.json",
    )


@lru_cache(maxsize=1)
def get_year_date_index():
    """Build a read-only date index from the existing ChromaDB collection."""

    return build_year_date_index(
        get_engine().collection
    )


# ============================================================
# PASSAGE IDENTITY
# ============================================================

def passage_key(chunk: dict) -> tuple[str, str, str]:

    return (
        str(chunk.get("speaker", "")).casefold(),
        str(chunk.get("date", "")),
        " ".join(
            str(chunk.get("text", "")).casefold().split()
        ),
    )


# ============================================================
# NORMALISE RETRIEVED CHUNKS
# ============================================================

def _clean_chunk(chunk: dict) -> dict:
    """Ensure evidence fields match the API and generator contracts."""

    result = dict(chunk)

    for field in (
        "speaker",
        "party",
        "date",
        "agenda",
        "text",
        "source",
    ):
        result[field] = str(
            result.get(field) or ""
        )

    try:
        result["rerank_score"] = float(
            result.get("rerank_score") or 0.0
        )

    except (ValueError, TypeError):
        result["rerank_score"] = 0.0

    if not result["source"]:
        result["source"] = "vector"

    return result


# ============================================================
# QUESTION TYPE
# ============================================================

def _temporal_type(question: str) -> str | None:

    q = question.casefold()

    if "customs union" in q:
        return "customs"

    if (
        "referendum" in q
        or "public vote" in q
    ):
        return "referendum"

    return None


# ============================================================
# TEMPORAL EVIDENCE SCORING
# ============================================================

def _temporal_strength(
    chunk: dict,
    topic: str,
) -> float:
    """
    Rank relevant source sentences, rather than
    accepting a generic speech match.
    """

    text = str(
        chunk.get("text", "")
    )

    agenda = str(
        chunk.get("agenda", "")
    )

    # Exclude unrelated Scottish independence debates.
    if topic == "referendum":

        if "claim of right for scotland" in agenda.casefold():
            return -100.0

        if SCOTTISH_TOPIC.search(text) and not re.search(
            r"brexit|european union|withdrawal agreement|eu referendum",
            text,
            re.I,
        ):
            return -100.0

    # Protect parliamentary honorifics before sentence splitting.
    protected = re.sub(
        r"\b(?:hon|Mr|Mrs|Ms|Dr|Prof|Rt)\.",
        lambda m: m.group(0)[:-1] + "<DOT>",
        text,
        flags=re.I,
    )

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z“‘])",
        protected,
    )

    best = -100.0

    for sentence in sentences:

        sentence = (
            sentence.replace("<DOT>", ".").strip()
        )

        if len(sentence) < 40:
            continue

        if topic == "customs":

            if "customs union" not in sentence.casefold():
                continue

            score = 7.0

            if CUSTOMS_CUE.search(sentence):
                score += 5.0

        elif topic == "referendum":

            if not VOTE_TOPIC.search(sentence):
                continue

            score = 7.0

            if VOTE_CUE.search(sentence):
                score += 5.0

        else:
            continue

        # Quality penalties.
        if "?" in sentence:
            score -= 1.5

        if "€" in sentence:
            score -= 1.0

        if sentence.casefold().startswith(
            ("i thank", "i am grateful", "let me now")
        ):
            score -= 6.0

        if len(sentence) > 500:
            score -= 2.0

        best = max(
            best,
            score,
        )

    return best


# ============================================================
# TARGETED TEMPORAL RETRIEVAL
# ============================================================

def _retrieve_temporal(
    engine,
    question: str,
    original: list[dict],
):
    """
    Retrieve evidence for each requested year.

    Preserve relevant original evidence.
    Search with several topic-specific formulations.
    Balance evidence across years and speakers.
    """

    match = YEAR_RANGE.search(question)

    if not match:
        return original, {
            "temporal_searches": 0,
            "weak_years": [],
        }

    start, end = map(
        int,
        match.groups(),
    )

    years = [
        str(year)
        for year in range(
            start,
            end + 1,
        )
    ]

    topic = _temporal_type(
        question
    )

    if topic is None:
        return original, {
            "temporal_searches": 0,
            "weak_years": [],
        }

    index = get_year_date_index()

    pools = {
        year: {}
        for year in years
    }

    # --------------------------------------------------------
    # ADD CANDIDATES WITHOUT DUPLICATES
    # --------------------------------------------------------

    def add(
        chunk: dict,
        year: str,
    ):

        if str(
            chunk.get("date", "")
        )[:4] != year:
            return

        if not (
            chunk.get("speaker")
            and chunk.get("text")
        ):
            return

        item = _clean_chunk(
            chunk
        )

        key = passage_key(
            item
        )

        prior = pools[year].get(
            key
        )

        if (
            prior is None
            or item["rerank_score"] > prior["rerank_score"]
        ):
            pools[year][key] = item

    # --------------------------------------------------------
    # PRESERVE EXISTING EVIDENCE
    # --------------------------------------------------------

    for chunk in original:

        year = str(
            chunk.get("date", "")
        )[:4]

        if year in pools:
            add(
                chunk,
                year,
            )

    # --------------------------------------------------------
    # TARGETED YEAR-CONSTRAINED SEARCHES
    # --------------------------------------------------------

    searches = 0

    for year in years:

        for query in TEMPORAL_QUERIES[topic]:

            try:

                candidates = retrieve_year_candidates(
                    engine=engine,
                    question=f"{query} {year}",
                    year=year,
                    dates_by_year=index,
                    n_results=35,
                )

                searches += 1

                for candidate in candidates:
                    add(
                        candidate,
                        year,
                    )

            except Exception:
                LOGGER.exception(
                    "Temporal search failed for %s",
                    year,
                )

    # --------------------------------------------------------
    # SELECT YEAR-BALANCED EVIDENCE
    # --------------------------------------------------------

    selected = []

    used = set()

    cross_party = (
        "cross-party"
        in question.casefold()
    )

    weak_years = []

    for year in years:

        scored = sorted(
            (
                (
                    _temporal_strength(
                        chunk,
                        topic,
                    )
                    + 0.10 * max(
                        -10.0,
                        min(
                            10.0,
                            chunk["rerank_score"],
                        ),
                    ),
                    chunk,
                )
                for chunk in pools[year].values()
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )

        if not scored:
            weak_years.append(year)
            continue

        if scored[0][0] < 10.0:
            weak_years.append(year)

        per_year = []

        # Up to 3 passages per year.
        for _ in range(3):

            eligible = []

            for score, chunk in scored:

                key = passage_key(
                    chunk
                )

                if key in used:
                    continue

                # Prefer different speakers within a year.
                if any(
                    chunk["speaker"] == previous["speaker"]
                    for previous in per_year
                ):
                    continue

                party_bonus = (
                    1.3
                    if (
                        cross_party
                        and not any(
                            previous["party"] == chunk["party"]
                            for previous in per_year
                        )
                    )
                    else 0.0
                )

                eligible.append(
                    (
                        score + party_bonus,
                        chunk,
                    )
                )

            if not eligible:
                break

            _, winner = max(
                eligible,
                key=lambda pair: pair[0],
            )

            per_year.append(
                winner
            )

            used.add(
                passage_key(winner)
            )

        selected.extend(
            per_year
        )

    # --------------------------------------------------------
    # FALLBACK TO ORIGINAL EVIDENCE
    # --------------------------------------------------------

    if not selected:

        LOGGER.warning(
            "No targeted temporal candidates; retaining baseline evidence."
        )

        return original, {
            "temporal_searches": searches,
            "weak_years": years,
        }

    # Sort once before citation IDs are assigned.
    selected.sort(
        key=lambda c: (
            c["date"],
            c["speaker"],
        )
    )

    return selected, {
        "temporal_searches": searches,
        "weak_years": weak_years,
    }


# ============================================================
# MAIN FASTAPI SERVICE FUNCTION
# ============================================================

def run_graphrag_question(question: str) -> dict:
    """
    Main function used by FastAPI.

    Preserves:
        question
        answer
        evidence
        retrieval
        coverage
    """

    if (
        not isinstance(question, str)
        or not question.strip()
    ):
        raise ValueError(
            "Question must be a non-empty string"
        )

    question = question.strip()

    # --------------------------------------------------------
    # STEP 1: VALIDATE QUESTION DATE RANGE
    # --------------------------------------------------------

    out_of_range = check_year_coverage(
        question
    )

    if out_of_range is not None:

        return {
            "question": question,
            "answer": (
                f"The speech database covers "
                f"{DATA_START_YEAR}–{DATA_END_YEAR}. "
                f"It cannot establish what Parliament "
                f"said in {out_of_range}."
            ),
            "evidence": [],
            "retrieval": {
                "r1_count": 0,
                "r2_count": 0,
                "final_count": 0,
                "r2_contributions": 0,
                "temporal_added": 0,
                "intent_added": 0,
            },
            "coverage": {
                "coverage_ok": False,
                "missing_years": [],
                "missing_parties": [],
            },
        }

    # --------------------------------------------------------
    # STEP 2: ORIGINAL GRAPHRAG EVIDENCE
    # --------------------------------------------------------

    engine = get_engine()

    base = engine.retrieve_iterative(
        question
    )

    baseline = [
        _clean_chunk(c)
        for c in base.get(
            "final_chunks",
            [],
        )
    ]

    temporal = (
        YEAR_RANGE.search(question)
        is not None
    )

    plan = {
        "kind": "general",
        "facets": [],
    }

    temporal_info = {
        "temporal_searches": 0,
        "weak_years": [],
    }

    # --------------------------------------------------------
    # STEP 3: QUESTION-AWARE OR TEMPORAL RETRIEVAL
    # --------------------------------------------------------

    if temporal:

        final_chunks, temporal_info = _retrieve_temporal(
            engine,
            question,
            baseline,
        )

    else:

        final_chunks, plan = retrieve_question_aware_evidence(
            engine=engine,
            question=question,
            baseline_chunks=baseline,
            max_chunks=10,
            n_results_per_search=30,
        )

    final_chunks = [
        _clean_chunk(c)
        for c in final_chunks
    ]

    if not temporal:
        final_chunks = final_chunks[:10]

    # --------------------------------------------------------
    # STEP 4: COVERAGE VALIDATION
    # --------------------------------------------------------

    coverage = check_evidence_coverage(
        question,
        final_chunks,
    )

    if not isinstance(coverage, dict):

        coverage = {
            "coverage_ok": bool(coverage),
            "missing_years": [],
            "missing_parties": [],
        }

    coverage = dict(
        coverage
    )

    coverage.update(
        {
            "question_intent": (
                "temporal"
                if temporal
                else plan.get("kind", "general")
            ),

            "facet_counts": plan.get(
                "facet_counts",
                {},
            ),

            "missing_facets": plan.get(
                "missing_facets",
                [],
            ),

            "requires_role_metadata": bool(
                plan.get(
                    "requires_role_metadata",
                    False,
                )
            ),
        }
    )

    # --------------------------------------------------------
    # STEP 5: RETRIEVAL DIAGNOSTICS
    # --------------------------------------------------------

    r1_keys = {
        passage_key(c)
        for c in base.get(
            "chunks_r1",
            [],
        )
    }

    r2_keys = {
        passage_key(c)
        for c in base.get(
            "chunks_r2",
            [],
        )
    }

    final_keys = {
        passage_key(c)
        for c in final_chunks
    }

    baseline_keys = {
        passage_key(c)
        for c in baseline
    }

    retrieval = {
        "r1_count": len(
            base.get("chunks_r1", [])
        ),

        "r2_count": len(
            base.get("chunks_r2", [])
        ),

        "final_count": len(
            final_chunks
        ),

        "r2_contributions": len(
            (r2_keys - r1_keys)
            & final_keys
        ),

        "temporal_added": (
            sum(
                1
                for c in final_chunks
                if passage_key(c) not in r1_keys
            )
            if temporal
            else 0
        ),

        "intent_added": (
            sum(
                1
                for c in final_chunks
                if passage_key(c) not in baseline_keys
            )
            if not temporal
            else 0
        ),

        "temporal_searches": temporal_info[
            "temporal_searches"
        ],

        "weak_years": temporal_info[
            "weak_years"
        ],
    }

    # --------------------------------------------------------
    # STEP 6: BUILD API EVIDENCE
    # --------------------------------------------------------

    evidence = [
        {
            key: c[key]
            for key in (
                "speaker",
                "party",
                "date",
                "agenda",
                "text",
                "source",
                "rerank_score",
            )
        }
        for c in final_chunks
    ]

    # --------------------------------------------------------
    # STEP 7: GENERATE GROUNDED ANSWER
    # --------------------------------------------------------

    if (
        not final_chunks
        or not coverage.get(
            "coverage_ok",
            True,
        )
    ):

        missing = []

        if coverage.get("missing_years"):

            missing.append(
                "years "
                + ", ".join(
                    coverage["missing_years"]
                )
            )

        if coverage.get("missing_parties"):

            missing.append(
                "parties "
                + ", ".join(
                    coverage["missing_parties"]
                )
            )

        answer = (
            "The selected speeches do not establish "
            "a sufficiently complete answer. "
            + (
                "Missing evidence for "
                + "; ".join(missing)
                + ". "
                if missing
                else ""
            )
            + "Relevant speeches may exist elsewhere "
            "in the database."
        )

    elif temporal:

        answer = generate_grounded_answer(
            engine,
            question,
            final_chunks,
        )

    else:

        answer = generate_general_grounded_answer(
            engine=engine,
            question=question,
            final_chunks=final_chunks,
            plan=plan,
        )

    # --------------------------------------------------------
    # STEP 8: RETURN UNCHANGED API CONTRACT
    # --------------------------------------------------------

    return {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "retrieval": retrieval,
        "coverage": coverage,
    }
