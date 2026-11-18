
"""
Corrected GraphRAG Engine
=========================

This module extends the original GraphRAG research engine
with improved entity extraction.

Improvements:
1. Prevent incorrect substring party matching.
2. Remove the erroneous backbench -> Conservative mapping.
3. Match speakers using complete names.
4. Prevent topic keywords from silently inserting parties.
5. Preserve the existing GraphRAG retrieval architecture.

The original src/graphrag_engine.py remains unchanged.
"""

from __future__ import annotations

import re

from src.graphrag_engine import (
    GraphRAGEngine as OriginalGraphRAGEngine,
)


# ============================================================
# CORRECTED GRAPHRAG ENGINE
# ============================================================

class GraphRAGEngine(OriginalGraphRAGEngine):
    """
    Production version of the original GraphRAG engine.

    Inherits:
        - Knowledge graph initialization
        - Graph traversal
        - ChromaDB retrieval
        - Sentence embeddings
        - Cross-encoder reranking
        - Iterative retrieval
        - HyDE expansion
        - Speaker profiles
        - Answer generation

    Overrides:
        - Entity extraction
    """

    # --------------------------------------------------------
    # POLITICAL PARTY ALIASES
    # --------------------------------------------------------

    PARTY_ALIASES = {
        "labour": "Lab",
        "labour party": "Lab",

        "conservative": "Con",
        "conservatives": "Con",
        "conservative party": "Con",
        "tory": "Con",
        "tories": "Con",

        "snp": "SNP",

        "lib dem": "LibDem",
        "lib dems": "LibDem",
        "liberal democrat": "LibDem",
        "liberal democrats": "LibDem",

        "dup": "DUP",

        "ukip": "UKIP",

        "plaid cymru": "PlaidCymru",

        "green party": "GPEW",
    }

    # ========================================================
    # STRICT PHRASE MATCHING
    # ========================================================

    @staticmethod
    def _has_phrase(
        text: str,
        phrase: str,
    ) -> bool:
        """
        Match complete words or phrases only.

        Examples:
            'labour' matches 'Labour MPs'.
            'con' does not match 'second'.
            'con' does not match 'economic'.
        """

        pattern = (
            r"(?<!\w)"
            + re.escape(phrase.casefold())
            + r"(?!\w)"
        )

        return bool(
            re.search(
                pattern,
                text.casefold(),
            )
        )

    # ========================================================
    # CORRECTED ENTITY EXTRACTION
    # ========================================================

    def extract_entities(
        self,
        text: str,
        top_k_topics: int = 15,
        similarity_threshold: float = 0.25,
    ) -> dict:
        """
        Extract speakers, political parties, years,
        and semantic topics from a question.

        Unlike the original implementation:
            - Backbench is not treated as a party.
            - Party codes use strict word matching.
            - Topic keywords do not automatically
              insert political parties or speakers.
        """

        query = str(text or "")

        q = query.casefold()

        # ----------------------------------------------------
        # STEP 1: MATCH EXPLICITLY NAMED SPEAKERS
        # ----------------------------------------------------

        matched_speakers = []

        for speaker in self.all_speakers:

            speaker_name = str(speaker)

            if self._has_phrase(
                q,
                speaker_name,
            ):

                matched_speakers.append(
                    speaker
                )

        # ----------------------------------------------------
        # STEP 2: MATCH EXPLICIT PARTY ALIASES
        # ----------------------------------------------------

        matched_parties = []

        for alias, party_code in (
            self.PARTY_ALIASES.items()
        ):

            if not self._has_phrase(
                q,
                alias,
            ):
                continue

            if party_code not in self.all_parties:
                continue

            if party_code not in matched_parties:

                matched_parties.append(
                    party_code
                )

        # ----------------------------------------------------
        # STEP 3: MATCH EXACT PARTY CODES
        # ----------------------------------------------------

        for party in self.all_parties:

            if self._has_phrase(
                q,
                str(party),
            ):

                if party not in matched_parties:

                    matched_parties.append(
                        party
                    )

        # ----------------------------------------------------
        # STEP 4: EXTRACT EXPLICIT YEARS
        # ----------------------------------------------------

        matched_years = []

        for year in self.all_years:

            if self._has_phrase(
                q,
                str(year),
            ):

                matched_years.append(
                    str(year)
                )

        # ----------------------------------------------------
        # STEP 5: EXPAND EXPLICIT YEAR RANGES
        # ----------------------------------------------------

        range_match = re.search(
            r"\b(?:between|from)\s+"
            r"(20\d{2})\s+"
            r"(?:and|to)\s+"
            r"(20\d{2})\b",
            q,
        )

        if range_match:

            start_year = int(
                range_match.group(1)
            )

            end_year = int(
                range_match.group(2)
            )

            if (
                2015 <= start_year
                <= end_year <= 2019
            ):

                for year in range(
                    start_year,
                    end_year + 1,
                ):

                    year_string = str(year)

                    if (
                        year_string in self.all_years
                        and year_string not in matched_years
                    ):

                        matched_years.append(
                            year_string
                        )

        # ----------------------------------------------------
        # STEP 6: SEMANTIC TOPIC MATCHING
        # ----------------------------------------------------

        # Reuse the original research implementation.
        # No changes are made to topic embeddings.

        topic_result = self.match_topics(
            query,
            top_k_topics=top_k_topics,
            similarity_threshold=similarity_threshold,
        )

        # ----------------------------------------------------
        # STEP 7: RETURN ENTITIES
        # ----------------------------------------------------

        return {
            "speakers": matched_speakers,

            "parties": matched_parties,

            "years": matched_years,

            "topics": (
                topic_result["topics"]
            ),

            "topic_sims": (
                topic_result["topic_sims"]
            ),
        }
