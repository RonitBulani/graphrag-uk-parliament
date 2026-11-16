
"""
GraphRAG FastAPI service layer.

This module connects the original GraphRAG research engine
to the FastAPI application.

Features
--------
1. Cached GraphRAG engine initialization.
2. Dataset year validation (2015-2019).
3. Original iterative GraphRAG retrieval.
4. Enhanced multi-query retrieval for temporal questions.
5. Evidence coverage checks.
6. Hybrid temporal answer generation.
7. Grounded general-purpose answer generation.
8. Citation-aligned evidence and retrieval diagnostics.

The original src/graphrag_engine.py is not modified.
"""

from __future__ import annotations

import logging
import re

from functools import lru_cache
from pathlib import Path


# ============================================================
# IMPORT ORIGINAL RESEARCH ENGINE
# ============================================================

from src.graphrag_engine import GraphRAGEngine


# ============================================================
# IMPORT APPLICATION-LEVEL RETRIEVAL
# ============================================================

from app.temporal_retrieval import (
    build_year_date_index,
)

from app.retrieval_upgrade import (
    retrieve_diverse_temporal_evidence,
)


# ============================================================
# IMPORT EVIDENCE VALIDATION
# ============================================================

from app.evidence_checks import (
    check_evidence_coverage,
)


# ============================================================
# IMPORT ANSWER GENERATORS
# ============================================================

from app.grounded_generation import (
    generate_grounded_answer,
)

from app.general_grounded_generation import (
    generate_general_grounded_answer,
)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_START_YEAR = 2015
DATA_END_YEAR = 2019

MAX_TEMPORAL_CHUNKS = 12

TEMPORAL_RESULTS_PER_SEARCH = 35

MIN_RERANK_SCORE = -1.5


# ============================================================
# TEMPORAL QUESTION DETECTION
# ============================================================

TEMPORAL_RANGE_PATTERN = re.compile(
    r"\b(?:between|from)\s+"
    r"(20\d{2})\s+"
    r"(?:and|to)\s+"
    r"(20\d{2})\b",
    flags=re.IGNORECASE,
)

SINGLE_YEAR_PATTERN = re.compile(
    r"\b(?:in|during)\s+(20\d{2})\b",
    flags=re.IGNORECASE,
)


# ============================================================
# YEAR VALIDATION
# ============================================================

def check_year_coverage(
    question: str,
) -> int | None:
    """
    Check whether a question explicitly requests
    parliamentary speeches outside the dataset period.

    Return the first unsupported year or None.
    """

    range_match = TEMPORAL_RANGE_PATTERN.search(
        question
    )

    if range_match:

        start_year = int(
            range_match.group(1)
        )

        end_year = int(
            range_match.group(2)
        )

        for year in (start_year, end_year):

            if (
                year < DATA_START_YEAR
                or year > DATA_END_YEAR
            ):
                return year

    single_match = SINGLE_YEAR_PATTERN.search(
        question
    )

    if single_match:

        year = int(
            single_match.group(1)
        )

        if (
            year < DATA_START_YEAR
            or year > DATA_END_YEAR
        ):
            return year

    return None


def get_requested_years(
    question: str,
) -> list[str]:
    """
    Extract a valid explicit year range.

    Examples
    --------
    between 2016 and 2019
    from 2016 to 2019

    Other questions return an empty list.
    """

    match = TEMPORAL_RANGE_PATTERN.search(
        question
    )

    if not match:
        return []

    start_year = int(
        match.group(1)
    )

    end_year = int(
        match.group(2)
    )

    if not (
        DATA_START_YEAR
        <= start_year
        <= end_year
        <= DATA_END_YEAR
    ):
        return []

    return [
        str(year)
        for year in range(
            start_year,
            end_year + 1,
        )
    ]


# ============================================================
# GRAPH RAG ENGINE INITIALIZATION
# ============================================================

@lru_cache(maxsize=1)
def get_engine():
    """
    Initialize the original GraphRAG engine once
    per Python process.

    Its graph, embedding model, reranker and
    ChromaDB collection are reused across requests.
    """

    graph_path = (
        PROJECT_ROOT
        / "data"
        / "knowledge_graph.graphml"
    )

    metadata_path = (
        PROJECT_ROOT
        / "data"
        / "knowledge_graph_metadata.json"
    )

    logger.info(
        "Initializing original GraphRAG engine."
    )

    engine = GraphRAGEngine(
        graph_path,
        metadata_path,
    )

    logger.info(
        "GraphRAG engine initialized."
    )

    return engine


# ============================================================
# YEAR-DATE INDEX
# ============================================================

@lru_cache(maxsize=1)
def get_year_date_index():
    """
    Build a read-only lookup of dates by year.

    This is used for strict ChromaDB year filtering.

    The vector database is not modified.
    """

    engine = get_engine()

    logger.info(
        "Building year-date index."
    )

    return build_year_date_index(
        engine.collection
    )


# ============================================================
# PASSAGE IDENTITY
# ============================================================

def passage_key(
    chunk: dict,
) -> tuple:
    """
    Identify the same speech passage across
    different retrieval stages.

    Retrieval source is intentionally excluded.
    """

    return (
        str(
            chunk.get("speaker", "")
        ).strip(),

        str(
            chunk.get("date", "")
        ).strip(),

        str(
            chunk.get("text", "")
        ).strip()[:160],
    )


# ============================================================
# STANDARD API EVIDENCE FORMAT
# ============================================================

def prepare_evidence(
    final_chunks: list[dict],
) -> list[dict]:
    """
    Convert retrieved passages to API evidence.

    IMPORTANT:
    Preserve final_chunks order.

    Answer citations [E1], [E2], etc. refer to
    this exact evidence array.
    """

    evidence = []

    for chunk in final_chunks:

        evidence.append(
            {
                "speaker": chunk.get(
                    "speaker", ""
                ),

                "party": chunk.get(
                    "party", ""
                ),

                "date": chunk.get(
                    "date", ""
                ),

                "agenda": chunk.get(
                    "agenda", ""
                ),

                "text": chunk.get(
                    "text", ""
                ),

                "source": chunk.get(
                    "source", ""
                ),

                "rerank_score": chunk.get(
                    "rerank_score", 0.0
                ),
            }
        )

    return evidence


# ============================================================
# RETRIEVAL DIAGNOSTICS
# ============================================================

def build_retrieval_diagnostics(
    result: dict,
    final_chunks: list[dict],
) -> dict:
    """
    Calculate actual retrieval contributions.

    Preserve the API fields used by the
    current frontend and acceptance checks.
    """

    chunks_r1 = result.get(
        "chunks_r1", []
    )

    chunks_r2 = result.get(
        "chunks_r2", []
    )

    r1_keys = {
        passage_key(chunk)
        for chunk in chunks_r1
    }

    r2_keys = {
        passage_key(chunk)
        for chunk in chunks_r2
    }

    r2_contributions = sum(
        1
        for chunk in final_chunks
        if (
            passage_key(chunk) in r2_keys
            and passage_key(chunk) not in r1_keys
        )
    )

    temporal_added = sum(
        1
        for chunk in final_chunks
        if str(
            chunk.get("source", "")
        ).startswith("year_filtered_")
    )

    return {
        "r1_count": len(
            chunks_r1
        ),

        "r2_count": len(
            chunks_r2
        ),

        "final_count": len(
            final_chunks
        ),

        "r2_contributions": (
            r2_contributions
        ),

        "temporal_added": (
            temporal_added
        ),
    }


# ============================================================
# EMPTY RETRIEVAL DIAGNOSTICS
# ============================================================

def empty_retrieval_diagnostics() -> dict:

    return {
        "r1_count": 0,
        "r2_count": 0,
        "final_count": 0,
        "r2_contributions": 0,
        "temporal_added": 0,
    }


# ============================================================
# MAIN GRAPHRAG API SERVICE
# ============================================================

def run_graphrag_question(
    question: str,
) -> dict:
    """
    Main service function called by FastAPI.

    Processing
    ----------
    1. Validate question and dataset coverage.
    2. Run the original iterative GraphRAG retrieval.
    3. Enhance explicit temporal-range retrieval.
    4. Prepare citation-aligned evidence.
    5. Calculate retrieval diagnostics.
    6. Check evidence coverage.
    7. Select the correct grounded generator.
    8. Return the complete API response.

    Original research code remains untouched.
    """

    # --------------------------------------------------------
    # STEP 1: VALIDATE QUESTION
    # --------------------------------------------------------

    question = str(
        question or ""
    ).strip()

    if not question:

        return {
            "question": question,

            "answer": (
                "Please provide a question "
                "about the parliamentary record."
            ),

            "evidence": [],

            "retrieval": (
                empty_retrieval_diagnostics()
            ),

            "coverage": {
                "coverage_ok": False,
                "requested_years": [],
                "evidence_years": [],
                "missing_years": [],
                "requested_parties": [],
                "missing_parties": [],
            },
        }

    # --------------------------------------------------------
    # STEP 2: CHECK DATASET YEAR COVERAGE
    # --------------------------------------------------------

    invalid_year = check_year_coverage(
        question
    )

    if invalid_year is not None:

        logger.warning(
            "Question requests unsupported year %s.",
            invalid_year,
        )

        return {
            "question": question,

            "answer": (
                "This database contains UK "
                "parliamentary Brexit speeches "
                f"from {DATA_START_YEAR} to "
                f"{DATA_END_YEAR}. "
                f"I cannot establish what "
                f"Parliament said in {invalid_year} "
                "using these records."
            ),

            "evidence": [],

            "retrieval": (
                empty_retrieval_diagnostics()
            ),

            "coverage": {
                "coverage_ok": False,
                "requested_years": [
                    str(invalid_year)
                ],
                "evidence_years": [],
                "missing_years": [
                    str(invalid_year)
                ],
                "requested_parties": [],
                "missing_parties": [],
            },
        }

    # --------------------------------------------------------
    # STEP 3: INITIALIZE GRAPH RAG ENGINE
    # --------------------------------------------------------

    engine = get_engine()

    # --------------------------------------------------------
    # STEP 4: ORIGINAL ITERATIVE RETRIEVAL
    # --------------------------------------------------------

    logger.info(
        "Starting original iterative GraphRAG retrieval."
    )

    result = engine.retrieve_iterative(
        question
    )

    # Copy the original retrieved passages.
    # Do not mutate the research engine output.

    final_chunks = list(
        result.get(
            "final_chunks", []
        ) or []
    )

    logger.info(
        "Original GraphRAG selected %d passages.",
        len(final_chunks),
    )

    # --------------------------------------------------------
    # STEP 5: DETECT TEMPORAL QUESTION
    # --------------------------------------------------------

    requested_years = get_requested_years(
        question
    )

    temporal_enhanced = False

    # --------------------------------------------------------
    # STEP 6: ENHANCED TEMPORAL RETRIEVAL
    # --------------------------------------------------------

    if requested_years:

        logger.info(
            "Temporal question detected: %s",
            requested_years,
        )

        # Cached, read-only ChromaDB date index.
        dates_by_year = get_year_date_index()

        # Multi-query, year-filtered retrieval.
        #
        # Uses the current app/retrieval_upgrade.py
        # and leaves the original engine unchanged.

        final_chunks = (
            retrieve_diverse_temporal_evidence(
                engine=engine,

                question=question,

                requested_years=(
                    requested_years
                ),

                dates_by_year=(
                    dates_by_year
                ),

                baseline_chunks=(
                    final_chunks
                ),

                max_chunks=(
                    MAX_TEMPORAL_CHUNKS
                ),

                n_results_per_search=(
                    TEMPORAL_RESULTS_PER_SEARCH
                ),

                min_score=(
                    MIN_RERANK_SCORE
                ),
            )
        )

        # Keep evidence in chronological order.
        #
        # The temporal generator uses this same
        # order to create citation IDs.

        final_chunks = sorted(
            final_chunks,
            key=lambda chunk: (
                str(
                    chunk.get("date", "")
                ),

                str(
                    chunk.get("speaker", "")
                ),
            ),
        )

        temporal_enhanced = True

        logger.info(
            "Enhanced temporal retrieval selected "
            "%d passages.",
            len(final_chunks),
        )

    # --------------------------------------------------------
    # STEP 7: PREPARE CITATION-ALIGNED EVIDENCE
    # --------------------------------------------------------

    evidence = prepare_evidence(
        final_chunks
    )

    # --------------------------------------------------------
    # STEP 8: CALCULATE RETRIEVAL DIAGNOSTICS
    # --------------------------------------------------------

    retrieval_info = (
        build_retrieval_diagnostics(
            result,
            final_chunks,
        )
    )

    # --------------------------------------------------------
    # STEP 9: CHECK EVIDENCE COVERAGE
    # --------------------------------------------------------

    coverage = check_evidence_coverage(
        question,
        final_chunks,
    )

    if not final_chunks:

        logger.warning(
            "No supporting speeches were retrieved."
        )

        return {
            "question": question,

            "answer": (
                "I could not retrieve sufficient "
                "parliamentary evidence to answer "
                "this question reliably."
            ),

            "evidence": evidence,

            "retrieval": retrieval_info,

            "coverage": coverage,
        }

    # --------------------------------------------------------
    # STEP 10: HANDLE INCOMPLETE EVIDENCE COVERAGE
    # --------------------------------------------------------

    if not coverage.get(
        "coverage_ok", True
    ):

        problems = []

        missing_years = coverage.get(
            "missing_years", []
        )

        missing_parties = coverage.get(
            "missing_parties", []
        )

        if missing_years:

            problems.append(
                "The selected evidence does "
                "not cover these requested years: "
                + ", ".join(missing_years)
                + "."
            )

        if missing_parties:

            problems.append(
                "The selected evidence does "
                "not cover these requested parties: "
                + ", ".join(missing_parties)
                + "."
            )

        if not problems:

            problems.append(
                "The available evidence does "
                "not meet the coverage requirements."
            )

        logger.warning(
            "Evidence coverage incomplete: %s",
            problems,
        )

        return {
            "question": question,

            "answer": (
                "I cannot reliably answer the full "
                "question using the selected evidence. "
                + " ".join(problems)
                + " Relevant passages may still "
                "exist in the database."
            ),

            "evidence": evidence,

            "retrieval": retrieval_info,

            "coverage": coverage,
        }

    # --------------------------------------------------------
    # STEP 11: GENERATE GROUNDED ANSWER
    # --------------------------------------------------------

    if temporal_enhanced:

        # Temporal questions:
        #
        # Use the existing hybrid generator with
        # chronological synthesis and NLI review.

        logger.info(
            "Using temporal grounded generation."
        )

        answer = generate_grounded_answer(
            engine,
            question,
            final_chunks,
        )

    else:

        # Non-temporal questions:
        #
        # Use the new general-purpose generator.
        #
        # This replaces the original uncited
        # engine.generate() API pathway.
        #
        # The original engine.generate() function
        # itself is preserved and not modified.

        logger.info(
            "Using general grounded generation."
        )

        answer = (
            generate_general_grounded_answer(
                engine,
                question,
                final_chunks,
            )
        )

    # --------------------------------------------------------
    # STEP 12: RETURN COMPLETE API RESPONSE
    # --------------------------------------------------------

    logger.info(
        "GraphRAG request completed. "
        "Temporal=%s Evidence=%d",
        temporal_enhanced,
        len(evidence),
    )

    return {
        "question": question,

        "answer": answer,

        "evidence": evidence,

        "retrieval": retrieval_info,

        "coverage": coverage,
    }
