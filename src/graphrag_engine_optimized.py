
"""Optimized, testable retrieval engine for UK Parliament GraphRAG.

Install as: src/graphrag_engine_optimized.py

This subclasses the existing corrected engine. It does NOT modify the
original experimental src/graphrag_engine.py or the persisted GraphML/ChromaDB.

Production default: a single retrieval pass without an interim
LLM answer. Set GRAPHRAG_ENABLE_R2=1 to enable the experimental second pass.

Key changes:
  * Larger retrieval candidates and final R1 pool (default 12).
  * Deduplication by full normalized passage, not speaker/date.
  * No sentiment interpreted as political stance in this ranking.
  * Explicit party coverage and per-speaker caps for diversity.
  * All stage source labels and existing API result keys retained.

The existing temporal year-constrained application retrieval is not replaced.
"""

from __future__ import annotations

import logging
import os
import re

from collections import Counter

from src.graphrag_engine_fixed import (
    GraphRAGEngine as CorrectedGraphRAGEngine,
)

from src.graphrag_engine import merge_entities


log = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION HELPERS
# ============================================================

def _env_int(
    name: str,
    default: int,
    low: int,
    high: int,
) -> int:

    try:
        return max(
            low,
            min(
                high,
                int(os.getenv(name, str(default))),
            ),
        )

    except ValueError:

        log.warning(
            "Invalid integer %s; using %d",
            name,
            default,
        )

        return default


def _enabled(
    name: str,
    default: bool = False,
) -> bool:

    value = os.getenv(
        name,
        "1" if default else "0",
    ).strip().casefold()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


# ============================================================
# PASSAGE IDENTITY
# ============================================================

def _text_key(c: dict) -> tuple[str, str, str]:
    """
    Identify an individual speech passage.

    Two different passages by the same speaker
    on the same date are not automatically duplicates.
    """

    return (
        str(
            c.get("speaker", "")
        ).casefold(),

        str(
            c.get("date", "")
        ),

        " ".join(
            str(c.get("text", "")).split()
        ).casefold(),
    )


# ============================================================
# DIVERSE FINAL SELECTION
# ============================================================

def _select_diverse(
    chunks: list[dict],
    limit: int,
    parties: list[str] | None = None,
) -> list[dict]:
    """
    Select relevant evidence without treating
    speaker/date as a duplicate.

    A maximum of two chunks per speaker is a
    diversity preference. It can be relaxed
    to fill the result set.
    """

    parties = list(
        dict.fromkeys(
            parties or []
        )
    )

    ordered = sorted(
        chunks,
        key=lambda x: float(
            x.get("rerank_score", -100)
        ),
        reverse=True,
    )

    selected: list[dict] = []

    used: set[tuple] = set()

    speaker_counts: Counter = Counter()

    def include(
        c: dict,
        enforce_speaker_cap: bool = True,
    ) -> bool:

        key = _text_key(c)

        if (
            key in used
            or len(selected) >= limit
        ):
            return False

        speaker = str(
            c.get("speaker", "")
        ).casefold()

        if (
            enforce_speaker_cap
            and speaker_counts[speaker] >= 2
        ):
            return False

        selected.append(c)

        used.add(key)

        speaker_counts[speaker] += 1

        return True

    # --------------------------------------------------------
    # COVER EXPLICITLY REQUESTED PARTIES
    # --------------------------------------------------------

    for party in parties:

        for c in ordered:

            if (
                c.get("party") == party
                and include(c)
            ):
                break

    # --------------------------------------------------------
    # FILL USING RANKED RELEVANCE
    # --------------------------------------------------------

    for c in ordered:
        include(c)

    # --------------------------------------------------------
    # RELAX SPEAKER CAP IF NECESSARY
    # --------------------------------------------------------

    if len(selected) < limit:

        for c in ordered:

            include(
                c,
                enforce_speaker_cap=False,
            )

    return selected


# ============================================================
# OPTIMIZED ENGINE
# ============================================================

class GraphRAGEngine(CorrectedGraphRAGEngine):
    """
    Compatibility-preserving optimized engine.

    Inherits:
        - Knowledge graph
        - Topic matching
        - Corrected entity extraction
        - ChromaDB
        - Embedding model
        - Cross-encoder reranker
        - Graph traversal

    Overrides:
        - Sentiment-based graph weighting
        - Retrieval candidate selection
        - Iterative retrieval orchestration
    """

    # ========================================================
    # DISABLE SENTIMENT-AS-STANCE IN GRAPH RANKING
    # ========================================================

    def _topic_sentiment_zscores(self, topic):
        """
        Sentiment polarity is not a reliable
        indicator of political stance.

        Returning an empty mapping makes the
        inherited traversal use neutral values.
        """

        return {}

    # ========================================================
    # ENHANCED GRAPHRAG RETRIEVAL
    # ========================================================

    def retrieve_graphrag(
        self,
        question,
        entities,
        graph_speakers,
        top_k_graph=20,
        top_k_vector=40,
        exclude_texts=None,
        extra_embedding=None,
    ):
        """
        Gather a broader set of candidates
        before reranking and final selection.

        Distinct passages by the same speaker
        on the same date are preserved.
        """

        excluded_prefixes = set(
            exclude_texts or ()
        )

        count = self.collection.count()

        if count <= 0:
            return []

        # ----------------------------------------------------
        # EMBED ORIGINAL QUESTION
        # ----------------------------------------------------

        question_emb = self.model.encode(
            [question]
        )

        # Optional experimental R2 embedding.
        if extra_embedding is not None:

            question_emb = (
                question_emb + extra_embedding
            ) / 2.0

        query_embedding = (
            question_emb.tolist()
        )

        discovered: dict[tuple, dict] = {}

        sources: Counter = Counter()

        # ----------------------------------------------------
        # SHARED CHROMADB SEARCH
        # ----------------------------------------------------

        def search(
            label: str,
            n_results: int,
            **filters,
        ) -> None:

            n_results = min(
                n_results,
                count,
            )

            if n_results < 1:
                return

            try:

                result = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=n_results,
                    include=[
                        "documents",
                        "metadatas",
                        "distances",
                    ],
                    **filters,
                )

            except Exception as exc:

                log.warning(
                    "Chroma %s search failed: %s",
                    label,
                    exc,
                )

                return

            documents = (
                result.get("documents") or [[]]
            )[0]

            metadatas = (
                result.get("metadatas") or [[]]
            )[0]

            distances = (
                result.get("distances") or [[]]
            )[0]

            for index, (
                document,
                metadata,
            ) in enumerate(
                zip(documents, metadatas)
            ):

                metadata = metadata or {}

                if (
                    not document
                    or not metadata.get("speaker")
                ):
                    continue

                if (
                    document.strip()[:80]
                    in excluded_prefixes
                ):
                    continue

                distance = (
                    float(distances[index])
                    if index < len(distances)
                    else 0.0
                )

                candidate = {
                    "text": document,

                    "speaker": metadata.get(
                        "speaker", ""
                    ),

                    "party": metadata.get(
                        "party", ""
                    ),

                    "date": metadata.get(
                        "date", ""
                    ),

                    "agenda": metadata.get(
                        "agenda", ""
                    ),

                    "distance": distance,

                    "source": label,
                }

                key = _text_key(
                    candidate
                )

                previous = discovered.get(
                    key
                )

                if (
                    previous is None
                    or distance < previous.get(
                        "distance",
                        float("inf"),
                    )
                ):

                    discovered[key] = candidate

                sources[label] += 1

        # ----------------------------------------------------
        # STAGE 1: GRAPH-SELECTED SPEAKERS
        # ----------------------------------------------------

        if graph_speakers:

            search(
                "graph",
                top_k_graph,
                where={
                    "speaker": {
                        "$in": list(graph_speakers)
                    }
                },
            )

        # ----------------------------------------------------
        # STAGE 2: GLOBAL VECTOR RETRIEVAL
        # ----------------------------------------------------

        search(
            "vector",
            max(40, top_k_vector),
        )

        # ----------------------------------------------------
        # STAGE 3: PARTY RETRIEVAL
        # ----------------------------------------------------

        for party in list(
            dict.fromkeys(
                entities.get("parties", [])
            )
        ):

            search(
                "party",
                24,
                where={
                    "party": {
                        "$eq": party
                    }
                },
            )

        # ----------------------------------------------------
        # STAGE 4: PINNED-SPEAKER RETRIEVAL
        # ----------------------------------------------------

        for speaker in list(
            dict.fromkeys(
                entities.get("speakers", [])
            )
        )[:6]:

            search(
                "pinned",
                18,
                where={
                    "speaker": {
                        "$eq": speaker
                    }
                },
            )

        # ----------------------------------------------------
        # STAGE 5: LEXICAL HYBRID RETRIEVAL
        # ----------------------------------------------------

        q_lower = question.casefold()

        keyword_terms = (
            "article 50",
            "backstop",
            "customs union",
            "single market",
            "second referendum",
        )

        for term in keyword_terms:

            if term in q_lower:

                search(
                    "keyword",
                    14,
                    where_document={
                        "$contains": term
                    },
                )

        # ----------------------------------------------------
        # STAGE 6: COLLECT UNIQUE CANDIDATES
        # ----------------------------------------------------

        candidates = list(
            discovered.values()
        )

        if not candidates:
            return []

        # ----------------------------------------------------
        # STAGE 7: CROSS-ENCODER RERANKING
        # ----------------------------------------------------

        pairs = [
            [
                question,
                c["text"],
            ]
            for c in candidates
        ]

        raw_scores = self.reranker.predict(
            pairs
        )

        # ----------------------------------------------------
        # STAGE 8: SMALL GRAPH DISCOVERY BOOST
        # ----------------------------------------------------

        chosen_speakers = {
            spk: index
            for index, spk in enumerate(
                graph_speakers or []
            )
        }

        total_graph = max(
            1,
            len(chosen_speakers),
        )

        for candidate, raw in zip(
            candidates,
            raw_scores,
        ):

            rank = chosen_speakers.get(
                candidate["speaker"]
            )

            graph_boost = (
                0.03 * (
                    1 - rank / total_graph
                )
                if rank is not None
                else 0.0
            )

            candidate[
                "rerank_score"
            ] = (
                float(raw)
                + graph_boost
            )

        # ----------------------------------------------------
        # STAGE 9: FINAL SELECTION
        # ----------------------------------------------------

        limit = _env_int(
            "GRAPHRAG_ENGINE_TOP_K",
            12,
            5,
            40,
        )

        final = _select_diverse(
            candidates,
            limit,
            entities.get("parties", []),
        )

        log.info(
            "Optimized R1 candidates=%d "
            "selected=%d sources=%s",
            len(candidates),
            len(final),
            dict(sources),
        )

        return final

    # ========================================================
    # SINGLE-PASS OR OPTIONAL ITERATIVE RETRIEVAL
    # ========================================================

    def retrieve_iterative(self, question):
        """
        Run R1 only by default.

        If GRAPHRAG_ENABLE_R2=1, optionally run
        the original interim-answer-based R2.

        Keeps the API response dictionary compatible
        with app/graphrag_service.py.
        """

        # ----------------------------------------------------
        # STEP 1: ENTITY EXTRACTION
        # ----------------------------------------------------

        entities = self.extract_entities(
            question
        )

        definitional = bool(
            re.match(
                r"^what is\b",
                question.strip().casefold(),
            )
        )

        # ----------------------------------------------------
        # STEP 2: GRAPH TRAVERSAL
        # ----------------------------------------------------

        if definitional:

            graph_speakers = []

            paths = {}

        else:

            graph_speakers, paths = (
                self.traverse_graph(
                    entities,
                    question,
                )
            )

        # ----------------------------------------------------
        # STEP 3: FIRST RETRIEVAL PASS
        # ----------------------------------------------------

        r1 = self.retrieve_graphrag(
            question,
            entities,
            graph_speakers,
        )

        r2: list[dict] = []

        graph_speakers_r2: list[str] = []

        interim = ""

        merged = entities

        # ----------------------------------------------------
        # STEP 4: OPTIONAL SECOND PASS
        # ----------------------------------------------------

        if (
            _enabled("GRAPHRAG_ENABLE_R2")
            and r1
            and not definitional
        ):

            try:

                interim = self.generate_interim(
                    question,
                    r1,
                )

                if interim:

                    new_entities = self.extract_entities(
                        interim
                    )

                    merged = merge_entities(
                        entities,
                        new_entities,
                    )

                    graph_speakers_r2, paths_r2 = (
                        self.traverse_graph(
                            merged,
                            question,
                        )
                    )

                    paths.update(
                        paths_r2
                    )

                    r2 = self.retrieve_graphrag(
                        question,
                        merged,
                        graph_speakers_r2,

                        exclude_texts={
                            c["text"].strip()[:80]
                            for c in r1
                        },

                        extra_embedding=self.model.encode(
                            [interim]
                        ),
                    )

            except Exception:

                log.exception(
                    "Optional R2 failed; "
                    "keeping R1 evidence."
                )

        # ----------------------------------------------------
        # STEP 5: MERGE AND SELECT
        # ----------------------------------------------------

        limit = _env_int(
            "GRAPHRAG_ENGINE_TOP_K",
            12,
            5,
            40,
        )

        final = _select_diverse(
            r1 + r2,
            limit,
            entities.get("parties", []),
        )

        # ----------------------------------------------------
        # STEP 6: RETRIEVAL DIAGNOSTICS
        # ----------------------------------------------------

        r1_keys = {
            _text_key(c)
            for c in r1
        }

        r2_added = sum(
            _text_key(c) not in r1_keys
            for c in final
        )

        # ----------------------------------------------------
        # STEP 7: API-COMPATIBLE RESULT
        # ----------------------------------------------------

        return {
            "question": question,

            "entities": entities,

            "merged_entities": merged,

            "graph_speakers": graph_speakers,

            "graph_speakers_r2": graph_speakers_r2,

            "speaker_paths": paths,

            "chunks_r1": r1,

            "chunks_r2": r2,

            "final_chunks": final,

            "interim_answer": interim,

            "r2_count": r2_added,

            "speaker_profiles": {},

            "engine_diagnostics": {
                "mode": (
                    "r1_plus_r2"
                    if r2
                    else "r1_only"
                ),

                "r2_enabled": _enabled(
                    "GRAPHRAG_ENABLE_R2"
                ),

                "r1_selected": len(r1),

                "r2_selected": len(r2),

                "r2_final_contributions": r2_added,

                "final_selected": len(final),
            },
        }
