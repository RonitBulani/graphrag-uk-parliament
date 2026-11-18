
"""
Question-Aware Retrieval — Version 2
====================================

Improves evidence selection for:
- Party comparisons
- Frontbench/backbench comparisons
- Arguments for leaving the EU
- Two-sided explanations
- Definition and process questions

Retrieval facets represent information needs,
not verified political positions.

The original GraphRAG engine, knowledge graph
and ChromaDB collection remain unchanged.
"""

from __future__ import annotations

import logging
import re

from collections import Counter, defaultdict


logger = logging.getLogger(__name__)


# ============================================================
# QUESTION FACET HELPER
# ============================================================

def _facet(
    name: str,
    title: str,
    query: str,
    party: str | None = None,
    extra_query: str | None = None,
) -> dict:

    return {
        "name": name,
        "title": title,
        "query": query,
        "party": party,
        "queries": (
            [query]
            + ([extra_query] if extra_query else [])
        ),
    }


# ============================================================
# QUESTION PLANNING
# ============================================================

def plan_question(question: str) -> dict:
    """Determine which types of evidence are needed."""

    q = " ".join(
        str(question or "").casefold().split()
    )

    labour = bool(
        re.search(r"\blabour\b", q)
    )

    conservative = bool(
        re.search(
            r"\b(?:conservative|conservatives|"
            r"tory|tories)\b",
            q,
        )
    )

    # --------------------------------------------------------
    # LABOUR FRONTBENCH / BACKBENCH
    # --------------------------------------------------------

    if (
        labour
        and re.search(r"\bbackbench\w*\b", q)
        and re.search(r"\bfrontbench\w*\b", q)
    ):

        return {
            "kind": "role_comparison",
            "requires_role_metadata": True,
            "facets": [
                _facet(
                    "labour_policy",
                    "Labour party policy statements",
                    (
                        "Labour frontbench shadow Brexit "
                        "secretary official policy second "
                        "referendum public vote "
                        "confirmatory ballot"
                    ),
                    party="Lab",
                    extra_query=(
                        "Labour policy our party would "
                        "support second referendum "
                        "Labour accepts referendum result"
                    ),
                ),

                _facet(
                    "labour_views",
                    "Labour MPs' differing views",
                    (
                        "Labour MPs I believe second "
                        "referendum people's vote "
                        "should be held or opposed"
                    ),
                    party="Lab",
                    extra_query=(
                        "Labour backbench MPs different "
                        "views on second Brexit referendum "
                        "no further vote final say"
                    ),
                ),
            ],
        }

    # --------------------------------------------------------
    # LABOUR / CONSERVATIVE COMPARISON
    # --------------------------------------------------------

    if (
        labour
        and conservative
        and re.search(
            r"\b(?:differ|different|compare|"
            r"contrast|versus|vs\.?|"
            r"position|stance)\b",
            q,
        )
    ):

        return {
            "kind": "party_comparison",
            "requires_role_metadata": False,
            "facets": [
                _facet(
                    "labour",
                    "Labour speeches",
                    (
                        "Labour's stated Brexit policy "
                        "we would customs union protect "
                        "jobs workers rights single market "
                        "referendum"
                    ),
                    party="Lab",
                    extra_query=(
                        "Labour party policy Brexit "
                        "withdrawal agreement close trading "
                        "relationship confirmatory referendum"
                    ),
                ),

                _facet(
                    "conservative",
                    "Conservative speeches",
                    (
                        "Conservative government Brexit "
                        "policy leaving customs union "
                        "single market independent trade "
                        "sovereignty"
                    ),
                    party="Con",
                    extra_query=(
                        "Conservative manifesto Brexit "
                        "deliver referendum result control "
                        "laws immigration withdrawal agreement"
                    ),
                ),
            ],
        }

    # --------------------------------------------------------
    # ARGUMENTS FOR LEAVING THE EU
    # --------------------------------------------------------

    leave_argument = bool(
        re.search(
            r"\b(?:arguments|case|reasons)\s+for\s+"
            r"(?:leaving|leave|brexit|withdrawal)\b",
            q,
        )
    ) or bool(
        re.search(
            r"\bwhy\b.*"
            r"\b(?:support|advocat\w*|vote(?:d)?)\b"
            r".*\b(?:leave|leaving|brexit)\b",
            q,
        )
    )

    if leave_argument:

        return {
            "kind": "arguments_for_leaving",
            "requires_role_metadata": False,
            "facets": [
                _facet(
                    "sovereignty",
                    "Sovereignty and democratic accountability",
                    (
                        "I support leaving the European "
                        "Union because Britain should take "
                        "back control of laws and sovereignty"
                    ),
                    extra_query=(
                        "2016 referendum why vote Leave EU "
                        "restore sovereign Parliament "
                        "democratic control legislation"
                    ),
                ),

                _facet(
                    "economic",
                    "Trade and economic arguments",
                    (
                        "I support Brexit leaving European "
                        "Union to negotiate independent "
                        "trade deals economic prosperity"
                    ),
                    extra_query=(
                        "2016 vote Leave European Union "
                        "new trade agreements outside EU "
                        "economic opportunity global trade"
                    ),
                ),

                _facet(
                    "borders",
                    "Borders and other arguments",
                    (
                        "I support leaving EU to control "
                        "immigration borders end free "
                        "movement EU budget contributions"
                    ),
                    extra_query=(
                        "2016 referendum why vote Leave "
                        "control immigration national "
                        "borders freedom of movement"
                    ),
                ),
            ],
        }

    # --------------------------------------------------------
    # NORTHERN IRELAND BACKSTOP
    # --------------------------------------------------------

    if (
        "backstop" in q
        and re.search(
            r"\b(?:why|controversial|disagree|"
            r"debate|critic\w*|oppos\w*)\b"
            r"|for and against",
            q,
        )
    ):

        return {
            "kind": "two_sided_explanation",
            "requires_role_metadata": False,
            "facets": [
                _facet(
                    "purpose",
                    "Purpose of the backstop",
                    (
                        "why Northern Ireland backstop "
                        "was necessary protect Good Friday "
                        "Belfast Agreement avoid hard border "
                        "peace"
                    ),
                    extra_query=(
                        "backstop needed if no agreed border "
                        "solution to prevent hard border "
                        "Ireland protect peace agreement"
                    ),
                ),

                _facet(
                    "objections",
                    "Arguments against the backstop",
                    (
                        "why Northern Ireland backstop "
                        "opposed DUP Union customs checks "
                        "Irish Sea regulatory alignment "
                        "sovereignty"
                    ),
                    extra_query=(
                        "backstop risks UK split Northern "
                        "Ireland Great Britain separate "
                        "rules indefinite arrangement "
                        "no unilateral exit"
                    ),
                ),
            ],
        }

    # --------------------------------------------------------
    # DEFINITION AND PROCESS
    # --------------------------------------------------------

    if (
        re.match(
            r"^(?:what is|what was|explain|define)\b",
            q,
        )
        and re.search(
            r"\b(?:how|why|triggered|"
            r"introduced|implemented)\b",
            q,
        )
    ):

        return {
            "kind": "definition_and_process",
            "requires_role_metadata": False,
            "facets": [
                _facet(
                    "definition",
                    "Definition",
                    question + " definition meaning legal mechanism",
                ),
                _facet(
                    "process",
                    "Process or consequences",
                    question + " steps dates notification procedure",
                ),
            ],
        }

    return {
        "kind": "general",
        "requires_role_metadata": False,
        "facets": [],
    }


# ============================================================
# SOFT EVIDENCE CUES
# ============================================================

# The following patterns affect ranking only.
# They must not be treated as authoritative
# classifications of any MP's political position.

_REPORTED_LEAVE = re.compile(
    r"\b(?:"
    r"the leave campaign(?:'s)?|"
    r"arguments? made (?:for|by)|"
    r"(?:we were|we are) told|"
    r"those who argued|"
    r"members who argued|"
    r"opponents might|"
    r"campaign claims?|"
    r"does (?:my|the) hon\.? friend not find|"
    r"we were promised"
    r")\b",
    re.IGNORECASE,
)

_CHALLENGED_LEAVE = re.compile(
    r"\b(?:"
    r"i find it remarkable|"
    r"find it curious|"
    r"instead|"
    r"and yet|"
    r"makes? a nonsense|"
    r"misleading|"
    r"contradiction|"
    r"not support his|"
    r"false claim|"
    r"i do not think we should leave|"
    r"stay in a reformed european union|"
    r"better off inside the european union"
    r")\b",
    re.IGNORECASE,
)

_DIRECT_LEAVE = re.compile(
    r"\b(?:"
    r"we should leave|"
    r"we must leave|"
    r"i (?:will|would) vote leave|"
    r"i support leaving|"
    r"if we left|"
    r"more prosperous if we left|"
    r"take back control of our own|"
    r"regaining control of our|"
    r"one of the key reasons "
    r"(?:for doing so|to leave)|"
    r"i want (?:us|britain|the uk) to leave|"
    r"we (?:will|would) "
    r"(?:be able to|have the ability to) "
    r"(?:negotiate|control)|"
    r"we (?:will|would) have an "
    r"independent trade policy|"
    r"important for us as we leave the eu"
    r")\b",
    re.IGNORECASE,
)

_POSITIVE_LEAVE = re.compile(
    r"\b(?:"
    r"independent trade policy|"
    r"control of (?:our own )?borders|"
    r"parliamentary sovereignty|"
    r"restore sovereignty|"
    r"end free movement|"
    r"end the jurisdiction|"
    r"free trade agreements|"
    r"our own laws|"
    r"take back control"
    r")\b",
    re.IGNORECASE,
)

_POLICY_OWN = re.compile(
    r"\b(?:"
    r"labour (?:would|will|accepts|"
    r"respects|wants|believes)|"
    r"labour'?s (?:approach|policy|position)|"
    r"our (?:party )?policy|"
    r"we (?:would|will) "
    r"(?:prioritise|deliver|seek|support)|"
    r"(?:our|the) manifesto "
    r"(?:said|promised|committed)|"
    r"conservative (?:party|government) "
    r"(?:will|would|is committed)"
    r")\b",
    re.IGNORECASE,
)

_ROLE_VIEW = re.compile(
    r"\b(?:"
    r"i (?:support|oppose|believe|want)|"
    r"we should (?:have|not have)|"
    r"i cannot support|"
    r"people should have a "
    r"(?:final|second) say|"
    r"feel utterly betrayed|"
    r"i would support"
    r")\b",
    re.IGNORECASE,
)

_BACKSTOP_OBJECTION = re.compile(
    r"\b(?:"
    r"backstop risks?|"
    r"risks? weakening|"
    r"rips up the union|"
    r"trapped in an arrangement|"
    r"rule taker|"
    r"no unilateral exit|"
    r"would split the united kingdom|"
    r"could split the united kingdom|"
    r"split the united kingdom|"
    r"checks on trade between northern "
    r"ireland and great britain|"
    r"separate (?:rules|arrangements) "
    r"for northern ireland|"
    r"barrier down the irish sea|"
    r"regulatory alignment.*"
    r"(?:control over|rips up)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

_BACKSTOP_PURPOSE = re.compile(
    r"\b(?:"
    r"need for a backstop|"
    r"backstop is (?:needed|required|there)|"
    r"backstop to (?:ensure|prevent|protect|kick in)|"
    r"backstop.*no hard border|"
    r"backstop.*peace process guarantee|"
    r"legally binding backstop|"
    r"backstop.*in the absence of agreed solutions"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)


# ============================================================
# PASSAGE IDENTITY
# ============================================================

def _identity(chunk: dict) -> tuple:
    """Identify a passage independently of retrieval source."""

    return (
        str(
            chunk.get("speaker", "")
        ).casefold(),

        str(
            chunk.get("date", "")
        ),

        " ".join(
            str(chunk.get("text", "")).split()
        ).casefold(),
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

def _valid(
    chunk: dict,
    party: str | None = None,
) -> bool:

    return (
        bool(
            str(chunk.get("speaker", "")).strip()
        )
        and bool(
            str(chunk.get("text", "")).strip()
        )
        and (
            party is None
            or chunk.get("party") == party
        )
    )


# ============================================================
# QUESTION-SPECIFIC RANKING
# ============================================================

def _score_hint(
    kind: str,
    facet: str,
    text: str,
) -> tuple[float, str]:
    """
    Return a soft ranking adjustment and diagnostic label.

    Labels such as possible_direct_advocacy are
    heuristic descriptions, not verified stance labels.
    """

    if kind == "arguments_for_leaving":

        reporting = bool(
            _REPORTED_LEAVE.search(text)
        )

        challenging = bool(
            _CHALLENGED_LEAVE.search(text)
        )

        direct = bool(
            _DIRECT_LEAVE.search(text)
        )

        relevant = bool(
            _POSITIVE_LEAVE.search(text)
        )

        if reporting and challenging:
            return -9.0, "reported_or_challenged"

        if reporting:
            return (
                -5.0 + (0.5 if relevant else 0.0),
                "reported_argument",
            )

        if challenging:
            return -7.0, "possibly_opposing"

        if direct:
            return (
                6.0 + (2.0 if relevant else 0.0),
                "possible_direct_advocacy",
            )

        return (
            1.5 if relevant else -2.0,
            "unclassified",
        )

    if kind == "two_sided_explanation":

        objection = bool(
            _BACKSTOP_OBJECTION.search(text)
        )

        purpose = bool(
            _BACKSTOP_PURPOSE.search(text)
        )

        if facet == "purpose":

            if objection and not purpose:
                return -10.0, "likely_objection"

            if purpose and not objection:
                return 8.0, "likely_purpose"

            return (
                -2.0 if objection else 0.0,
                "mixed_or_uncertain",
            )

        if facet == "objections":

            if objection and not purpose:
                return 8.0, "likely_objection"

            if purpose and not objection:
                return -10.0, "likely_purpose"

            return (
                -2.0 if purpose else 0.0,
                "mixed_or_uncertain",
            )

    if kind == "party_comparison":

        if _POLICY_OWN.search(text):
            return 5.0, "possible_policy_statement"

        return 0.0, "other"

    if kind == "role_comparison":

        if facet == "labour_policy":

            if _POLICY_OWN.search(text):
                return 5.0, "possible_party_policy"

            return 0.0, "other"

        if _ROLE_VIEW.search(text):
            return 4.0, "possible_personal_view"

        return 0.0, "other"

    return 0.0, "unclassified"


# ============================================================
# CHROMADB RETRIEVAL
# ============================================================

def _search(
    engine,
    query: str,
    party: str | None,
    n_results: int,
) -> list[dict]:
    """
    Search the existing ChromaDB collection.

    No collection changes or reindexing take place.
    """

    collection_size = (
        engine.collection.count()
    )

    if collection_size == 0:
        return []

    query_embedding = (
        engine.model.encode([query]).tolist()
    )

    arguments = {
        "query_embeddings": query_embedding,
        "n_results": min(
            n_results,
            collection_size,
        ),
    }

    if party:

        arguments["where"] = {
            "party": {
                "$eq": party
            }
        }

    results = engine.collection.query(
        **arguments
    )

    documents = (
        results.get("documents")
        or [[]]
    )[0]

    metadatas = (
        results.get("metadatas")
        or [[]]
    )[0]

    chunks = []

    for document, metadata in zip(
        documents,
        metadatas,
    ):

        metadata = metadata or {}

        chunks.append(
            {
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

                "speech_index": metadata.get(
                    "speech_index"
                ),

                "source": "question_aware",
            }
        )

    return chunks


# ============================================================
# NEAR-DUPLICATE DETECTION
# ============================================================

def _overlaps_existing(
    candidate: dict,
    selected: list[dict],
) -> bool:
    """
    Identify highly overlapping chunks from
    the same speaker on the same date.

    Distinct statements remain eligible.
    """

    speaker = str(
        candidate.get("speaker", "")
    ).casefold()

    date = str(
        candidate.get("date", "")
    )

    words = set(
        re.findall(
            r"\w+",
            str(
                candidate.get("text", "")
            ).casefold(),
        )
    )

    for previous in selected:

        if (
            str(
                previous.get("speaker", "")
            ).casefold()
            != speaker
        ):
            continue

        if str(
            previous.get("date", "")
        ) != date:
            continue

        previous_words = set(
            re.findall(
                r"\w+",
                str(
                    previous.get("text", "")
                ).casefold(),
            )
        )

        if not words or not previous_words:
            continue

        overlap = (
            len(words & previous_words)
            / len(words | previous_words)
        )

        if overlap >= 0.72:
            return True

    return False


# ============================================================
# MAIN QUESTION-AWARE RETRIEVAL
# ============================================================

def retrieve_question_aware_evidence(
    engine,
    question: str,
    baseline_chunks: list[dict],
    max_chunks: int = 10,
    n_results_per_search: int = 30,
) -> tuple[list[dict], dict]:
    """
    Retrieve evidence based on question requirements.

    Steps:
    1. Plan answer facets.
    2. Preserve original GraphRAG candidates.
    3. Perform targeted semantic searches.
    4. Deduplicate.
    5. Rerank and apply soft evidence-quality hints.
    6. Balance selections across facets.

    The source's retrieval facet does not constitute
    a verified classification of its political stance.
    """

    # --------------------------------------------------------
    # STEP 1: QUESTION PLAN
    # --------------------------------------------------------

    plan = plan_question(question)

    facets = plan["facets"]

    if not facets:

        return (
            list(baseline_chunks or []),
            plan,
        )

    if max_chunks < 1:
        return [], plan

    # --------------------------------------------------------
    # STEP 2: BUILD CANDIDATE POOLS
    # --------------------------------------------------------

    pools = defaultdict(list)

    for facet in facets:

        name = facet["name"]

        party = facet.get("party")

        candidates = {}

        # Preserve suitable original retrieval results.

        for item in baseline_chunks or []:

            if _valid(item, party):

                candidates.setdefault(
                    _identity(item),
                    dict(item),
                )

        # ----------------------------------------------------
        # STEP 3: TARGETED SEARCHES
        # ----------------------------------------------------

        queries = (
            facet.get("queries")
            or [facet["query"]]
        )

        results_per_query = max(
            8,
            n_results_per_search
            // max(1, len(queries)),
        )

        for query in queries:

            try:

                discovered = _search(
                    engine,
                    query,
                    party,
                    results_per_query,
                )

                for item in discovered:

                    if _valid(item, party):

                        candidates.setdefault(
                            _identity(item),
                            item,
                        )

            except Exception:

                logger.exception(
                    "Question-aware search failed "
                    "for facet %s",
                    name,
                )

        rows = list(
            candidates.values()
        )

        if not rows:
            continue

        # ----------------------------------------------------
        # STEP 4: CROSS-ENCODER RERANKING
        # ----------------------------------------------------

        try:

            pairs = [
                [
                    facet["query"],
                    chunk["text"],
                ]
                for chunk in rows
            ]

            relevance = (
                engine.reranker.predict(
                    pairs
                )
            )

        except Exception:

            logger.exception(
                "Reranking failed for facet %s",
                name,
            )

            relevance = [
                chunk.get(
                    "rerank_score",
                    -10.0,
                )
                for chunk in rows
            ]

        # ----------------------------------------------------
        # STEP 5: SOFT QUALITY ADJUSTMENT
        # ----------------------------------------------------

        for item, score in zip(
            rows,
            relevance,
        ):

            candidate = dict(
                item
            )

            raw_score = float(
                score
            )

            bonus, hint_label = _score_hint(
                plan["kind"],
                name,
                str(
                    candidate["text"]
                ),
            )

            candidate[
                "rerank_score"
            ] = raw_score

            candidate[
                "retrieval_facet"
            ] = name

            candidate[
                "_selection_score"
            ] = raw_score + bonus

            candidate[
                "_source_type_hint"
            ] = hint_label

            pools[name].append(
                candidate
            )

        pools[name].sort(
            key=lambda item: item[
                "_selection_score"
            ],
            reverse=True,
        )

    # --------------------------------------------------------
    # STEP 6: INITIALIZE SELECTION
    # --------------------------------------------------------

    selected = []

    used = set()

    by_speaker = Counter()

    by_facet = Counter()

    maximum_per_speaker = 2

    # --------------------------------------------------------
    # STEP 7: SELECT FROM FACET
    # --------------------------------------------------------

    def pick(
        name: str,
        facet_limit: int,
    ) -> bool:

        if len(selected) >= max_chunks:
            return False

        if by_facet[name] >= facet_limit:
            return False

        for item in pools[name]:

            identity = _identity(
                item
            )

            speaker = str(
                item["speaker"]
            ).casefold()

            if identity in used:
                continue

            if (
                by_speaker[speaker]
                >= maximum_per_speaker
            ):
                continue

            if _overlaps_existing(
                item,
                selected,
            ):
                continue

            chosen = dict(
                item
            )

            chosen.pop(
                "_selection_score",
                None,
            )

            selected.append(
                chosen
            )

            used.add(
                identity
            )

            by_speaker[speaker] += 1

            by_facet[name] += 1

            return True

        return False

    # --------------------------------------------------------
    # STEP 8: GUARANTEE INITIAL FACET REPRESENTATION
    # --------------------------------------------------------

    for facet in facets:

        pick(
            facet["name"],
            max_chunks,
        )

    # --------------------------------------------------------
    # STEP 9: BALANCE SELECTION
    # --------------------------------------------------------

    target_per_facet = max(
        2,
        max_chunks // len(facets),
    )

    for _ in range(
        target_per_facet
    ):

        for facet in facets:

            pick(
                facet["name"],
                target_per_facet,
            )

    # --------------------------------------------------------
    # STEP 10: FILL AVAILABLE CAPACITY
    # --------------------------------------------------------

    while len(selected) < max_chunks:

        changed = False

        for facet in facets:

            changed = (
                pick(
                    facet["name"],
                    max_chunks,
                )
                or changed
            )

        if not changed:
            break

    # --------------------------------------------------------
    # STEP 11: RECORD DIAGNOSTICS
    # --------------------------------------------------------

    plan = dict(
        plan
    )

    plan["facet_counts"] = {
        facet["name"]: (
            by_facet[
                facet["name"]
            ]
        )
        for facet in facets
    }

    plan["missing_facets"] = [
        facet["name"]
        for facet in facets
        if (
            by_facet[
                facet["name"]
            ] == 0
        )
    ]

    plan["source_type_hints"] = dict(
        Counter(
            str(
                chunk.get(
                    "_source_type_hint",
                    "unclassified",
                )
            )
            for chunk in selected
        )
    )

    logger.info(
        "Question-aware retrieval: "
        "intent=%s, selected=%d, "
        "facets=%s, source_hints=%s",
        plan["kind"],
        len(selected),
        plan["facet_counts"],
        plan["source_type_hints"],
    )

    # --------------------------------------------------------
    # STEP 12: RETURN
    # --------------------------------------------------------

    return (
        selected,
        plan,
    )
