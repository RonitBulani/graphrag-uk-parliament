
"""Application-level multi-query temporal retrieval.

Uses the existing year-filtered ChromaDB retriever.
Search facet names are discovery routes, not verified
speaker stance labels.

The original GraphRAG engine is not modified.
"""

from __future__ import annotations

import math
import re

from collections import defaultdict
from typing import Any

from app.temporal_retrieval import (
    retrieve_year_candidates,
)


_SECOND_VOTE = re.compile(
    r"\b(?:second|another|further)\s+"
    r"(?:brexit\s+)?referendum\b"
    r"|\bpeople['’]?s\s+vote\b"
    r"|\bconfirmatory\s+(?:public\s+)?"
    r"(?:vote|ballot|referendum)\b",
    re.I,
)

_VOTE_TERMS = re.compile(
    r"\breferendum\b"
    r"|\bpublic vote\b"
    r"|\bpeople['’]?s vote\b"
    r"|\bconfirmatory (?:vote|ballot)\b"
    r"|\bfinal say\b"
    r"|\bdecision referendum\b",
    re.I,
)

_INDEPENDENCE_AGENDA = re.compile(
    r"\b(?:scottish independence|"
    r"independence referendum|"
    r"referendum on scottish independence)\b",
    re.I,
)

_EXPLICIT_BREXIT_VOTE = re.compile(
    r"\b(?:second|another|further) "
    r"brexit referendum\b"
    r"|\bpeople['’]?s vote\b"
    r"|\bconfirmatory (?:public )?"
    r"(?:vote|ballot)\b"
    r"|\bvote on (?:the )?"
    r"(?:brexit|withdrawal) deal\b",
    re.I,
)

# Used only for candidate ranking.
# A match does NOT verify a speaker's stance.
_PROPOSAL_CUES = re.compile(
    r"\b(?:decision referendum|"
    r"people['’]?s vote|"
    r"confirmatory ballot|"
    r"public vote on|"
    r"vote on the (?:final )?"
    r"(?:brexit )?deal|"
    r"referendum on the terms|"
    r"give the people a final say|"
    r"have their say over what comes next|"
    r"should have (?:a|the) "
    r"(?:public )?vote)\b",
    re.I,
)


# ============================================================
# PASSAGE IDENTIFICATION
# ============================================================

def _key(chunk: dict) -> tuple:

    return (
        str(
            chunk.get("speaker", "")
        ).strip().casefold(),

        str(
            chunk.get("date", "")
        )[:10],

        " ".join(
            str(
                chunk.get("text", "")
            ).split()
        ).casefold(),
    )


# ============================================================
# BASIC RELEVANCE CHECKS
# ============================================================

def _valid(
    chunk: dict,
    year: str,
    second_vote: bool,
) -> bool:

    date = str(
        chunk.get("date", "")
    )

    text = str(
        chunk.get("text", "")
    )

    speaker = str(
        chunk.get("speaker", "")
    )

    if not date.startswith(
        year + "-"
    ):
        return False

    if not speaker.strip() or not text.strip():
        return False

    if not second_vote:
        return True

    # Exclude speeches before the original
    # June 2016 Brexit referendum.
    if date < "2016-06-24":
        return False

    # Exclude debates whose main agenda is
    # Scottish independence.
    if _INDEPENDENCE_AGENDA.search(
        str(chunk.get("agenda", ""))
    ):
        return False

    if not _VOTE_TERMS.search(text):
        return False

    # If independence is discussed in the text,
    # require explicit reference to a Brexit vote.
    if _INDEPENDENCE_AGENDA.search(text):

        if not _EXPLICIT_BREXIT_VOTE.search(text):
            return False

    return True


# ============================================================
# YEAR-SPECIFIC SEARCH EXPANSION
# ============================================================

def _facets(
    question: str,
    year: str,
) -> dict[str, str]:

    facets = {
        "original": question,
    }

    if not _SECOND_VOTE.search(question):
        return facets

    # Search for terminology used in the earlier
    # stages of the Brexit debate.
    if int(year) <= 2017:

        facets["early_public_vote"] = (
            "MP proposing a decision referendum "
            "on the terms of Brexit, a public vote "
            "after EU negotiations, or the right "
            "of voters to decide whether to accept "
            "the negotiated withdrawal deal."
        )

    # Search for public-vote arguments.
    facets["public_vote"] = (
        "Parliamentary argument for a second "
        "Brexit referendum, People's Vote, "
        "final public say, or a confirmatory "
        "ballot on the negotiated Brexit "
        "withdrawal agreement."
    )

    # Search for objections to another vote.
    facets["against_vote"] = (
        "Parliamentary argument rejecting a "
        "second Brexit referendum or People's "
        "Vote because it would reverse the "
        "2016 result, prolong uncertainty, "
        "or undermine democratic trust."
    )

    return facets


# ============================================================
# CANDIDATE RANKING
# ============================================================

def _rank(
    chunk: dict,
    facet: str,
) -> float:

    raw_score = chunk.get(
        "rerank_score"
    )

    score = (
        float(raw_score)
        if isinstance(raw_score, (int, float))
        else -100.0
    )

    # A small relevance boost for terminology
    # associated with a further public vote.
    #
    # This is NOT a stance classification.
    if facet in {
        "early_public_vote",
        "public_vote",
    }:

        if _PROPOSAL_CUES.search(
            str(chunk.get("text", ""))
        ):
            score += 1.0

    return score


# ============================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================

def retrieve_diverse_temporal_evidence(
    *,
    engine: Any,
    question: str,
    requested_years: list[str],
    dates_by_year: dict,
    baseline_chunks: list[dict] | None = None,
    max_chunks: int = 12,
    n_results_per_search: int = 35,
    min_score: float = -1.5,
) -> list[dict]:

    """
    Retrieve year-correct and speaker-diverse evidence.

    Steps:
    1. Preserve original GraphRAG evidence.
    2. Run multiple year-filtered searches.
    3. Remove duplicate passages.
    4. Require correct years.
    5. Select from different search pathways.
    6. Limit repeated speakers within a year.
    7. Return chronological evidence.

    Search pathways do not establish political stance.
    """

    years = list(
        dict.fromkeys(
            map(str, requested_years)
        )
    )

    if not years:
        return []

    if max_chunks < len(years):
        raise ValueError(
            "max_chunks must be at least "
            "the number of requested years."
        )

    second_vote = bool(
        _SECOND_VOTE.search(question)
    )

    pools = {
        year: defaultdict(list)
        for year in years
    }

    routes = {
        year: _facets(question, year)
        for year in years
    }

    # --------------------------------------------------------
    # STEP 1: PRESERVE ORIGINAL GRAPHRAG EVIDENCE
    # --------------------------------------------------------

    for raw in baseline_chunks or []:

        year = str(
            raw.get("date", "")
        )[:4]

        if year not in pools:
            continue

        if _valid(
            raw,
            year,
            second_vote,
        ):

            pools[year]["baseline"].append(
                dict(raw)
            )

    # --------------------------------------------------------
    # STEP 2: MULTI-QUERY YEAR-FILTERED RETRIEVAL
    # --------------------------------------------------------

    for year in years:

        if not dates_by_year.get(year):
            continue

        for facet, query in routes[year].items():

            found = retrieve_year_candidates(
                engine=engine,
                question=query,
                year=year,
                dates_by_year=dates_by_year,
                n_results=n_results_per_search,
            )

            seen = set()

            for candidate in found:

                if not _valid(
                    candidate,
                    year,
                    second_vote,
                ):
                    continue

                key = _key(candidate)

                if key in seen:
                    continue

                seen.add(key)

                entry = dict(candidate)

                entry["retrieval_facet"] = facet

                pools[year][facet].append(
                    entry
                )

            pools[year][facet].sort(
                key=lambda item: _rank(
                    item,
                    facet,
                ),
                reverse=True,
            )

    # --------------------------------------------------------
    # STEP 3: PREPARE DIVERSITY TRACKING
    # --------------------------------------------------------

    selected = []

    used_passages = set()

    used_speaker_years = set()

    counts = defaultdict(int)

    per_year_cap = math.ceil(
        max_chunks / len(years)
    )

    def add_one(
        year: str,
        route: str,
    ) -> bool:

        if len(selected) >= max_chunks:
            return False

        if counts[year] >= per_year_cap:
            return False

        for chunk in pools[year].get(route, []):

            raw = chunk.get(
                "rerank_score"
            )

            if isinstance(raw, (int, float)):
                if raw < min_score:
                    continue

            key = _key(chunk)

            speaker_year = (
                key[0],
                year,
            )

            if key in used_passages:
                continue

            if speaker_year in used_speaker_years:
                continue

            selected.append(chunk)

            used_passages.add(key)

            used_speaker_years.add(
                speaker_year
            )

            counts[year] += 1

            return True

        return False

    # --------------------------------------------------------
    # STEP 4: ENSURE YEAR COVERAGE
    # --------------------------------------------------------

    for year in years:

        if not add_one(
            year,
            "original",
        ):

            add_one(
                year,
                "baseline",
            )

    # --------------------------------------------------------
    # STEP 5: FIND PUBLIC-VOTE ARGUMENTS
    # --------------------------------------------------------

    for year in years:

        preferred = (
            "early_public_vote"
            if "early_public_vote" in routes[year]
            else "public_vote"
        )

        if not add_one(
            year,
            preferred,
        ):

            add_one(
                year,
                "public_vote",
            )

    # --------------------------------------------------------
    # STEP 6: FIND CONTRASTING ARGUMENTS
    # --------------------------------------------------------

    for year in years:

        add_one(
            year,
            "against_vote",
        )

    # --------------------------------------------------------
    # STEP 7: FILL REMAINING CAPACITY
    # --------------------------------------------------------

    while len(selected) < max_chunks:

        changed = False

        for year in years:

            for route in [
                *routes[year],
                "baseline",
            ]:

                if add_one(
                    year,
                    route,
                ):

                    changed = True
                    break

        if not changed:
            break

    # --------------------------------------------------------
    # STEP 8: CHRONOLOGICAL ORDER
    # --------------------------------------------------------

    return sorted(
        selected,
        key=lambda item: (
            str(item.get("date", "")),
            str(item.get("speaker", "")),
        ),
    )
