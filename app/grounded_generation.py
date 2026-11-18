
"""Conservative hybrid temporal synthesis for GraphRAG.

- Generates atomic claims with the configured Ollama model.
- Checks claims against original source windows using NLI.
- Uses verbatim quotations when claims are not accepted.
- Builds a chronological explanation without another
  unrestricted LLM generation pass.
- Keeps the original GraphRAG engine unchanged.

NLI is a statistical safeguard, not a proof of correctness.
"""

from __future__ import annotations

import json
import logging
import os
import re

from collections import defaultdict
from functools import lru_cache


logger = logging.getLogger(__name__)

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

MIN_ENTAILMENT = 0.88
MAX_CONTRADICTION = 0.08


_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
    r"|(?<=[.!?])(?=[A-Z])"
)

_VOTE = re.compile(
    r"second referendum|"
    r"another referendum|"
    r"further referendum|"
    r"people['’]?s vote|"
    r"confirmatory (?:ballot|vote)|"
    r"public vote|"
    r"decision referendum|"
    r"three-way referendum|"
    r"new referendum|"
    r"final say",
    re.I,
)

_OWN_LANGUAGE = re.compile(
    r"\b(?:"
    r"I (?:regret|oppose|support|believe|"
    r"think|want|urge)|"
    r"we (?:are not calling|support|oppose|"
    r"believe|want|need)|"
    r"there (?:should|must) be|"
    r"Labour would support|"
    r"a second referendum (?:would|is)|"
    r"to hold a second referendum would"
    r")\b",
    re.I,
)

_REPORTED = re.compile(
    r"\b(?:petition|Nigel Farage|"
    r"he said|they said|those who|"
    r"some politicians|former Prime Minister)\b",
    re.I,
)


# ============================================================
# SOURCE SEGMENTATION
# ============================================================

def _segments(source: str) -> list[str]:

    source = str(
        source or ""
    )

    parts = [
        part.strip()
        for part in _SENTENCE_BOUNDARY.split(source)
    ]

    return [
        part
        for part in parts
        if part and part in source
    ]


# ============================================================
# RELEVANCE SCORING FOR QUOTES
# ============================================================

def _score(span: str) -> float:

    return (
        (
            7.0
            if _VOTE.search(span)
            else 0.0
        )
        +
        (
            9.0
            if _OWN_LANGUAGE.search(span)
            else 0.0
        )
        -
        (
            4.0
            if _REPORTED.search(span)
            else 0.0
        )
        -
        (
            4.0
            if len(span) < 28
            else 0.0
        )
        -
        (
            7.0
            if not _VOTE.search(span)
            else 0.0
        )
        -
        (
            5.0
            if re.search(
                r"\b(?:risks of that|"
                r"such a proposal|this issue)\b",
                span,
                re.I,
            )
            else 0.0
        )
    )


# ============================================================
# EXACT SOURCE QUOTATION
# ============================================================

def _quote(
    chunk: dict,
    limit: int = 330,
) -> str:

    source = str(
        chunk.get("text", "")
    )

    sentences = _segments(source)

    if not sentences:
        return ""

    best = max(
        sentences,
        key=_score,
    )

    if len(best) <= limit:
        return best

    anchor = _VOTE.search(best)

    start = (
        max(0, anchor.start() - 85)
        if anchor
        else 0
    )

    if start:

        next_space = best.find(
            " ",
            start,
        )

        if next_space != -1:
            start = next_space + 1

    end = min(
        len(best),
        start + limit,
    )

    if end < len(best):

        previous_space = best.rfind(
            " ",
            start,
            end,
        )

        if previous_space > start:
            end = previous_space

    snippet = best[start:end]

    if snippet not in source:
        raise ValueError(
            "Quoted text is not present "
            "in the original speech."
        )

    return (
        ("…" if start else "")
        + snippet
        + ("…" if end < len(best) else "")
    )


# ============================================================
# CONTEXT WINDOWS FOR NLI
# ============================================================

def _windows(
    source: str,
    count: int = 3,
) -> dict[str, str]:

    """
    Build source-owned windows with neighbouring
    sentences where possible.
    """

    cuts = (
        [0]
        + [
            match.end()
            for match in _SENTENCE_BOUNDARY.finditer(
                source
            )
        ]
        + [len(source)]
    )

    positions = [
        (a, b)
        for a, b in zip(cuts, cuts[1:])
        if source[a:b].strip()
    ]

    sentences = [
        source[a:b].strip()
        for a, b in positions
    ]

    order = sorted(
        range(len(sentences)),
        key=lambda i: (
            -_score(sentences[i]),
            i,
        ),
    )

    selected = {}
    seen = set()

    for i in order:

        start = positions[
            max(0, i - 1)
        ][0]

        end = positions[
            min(
                len(sentences) - 1,
                i + 1,
            )
        ][1]

        window = source[
            start:end
        ].strip()

        if len(window) > 1100:
            window = sentences[i]

        if len(window) > 1100:

            window = (
                window[:1100]
                .rsplit(" ", 1)[0]
            )

        if not window:
            continue

        if window in seen:
            continue

        if window not in source:
            continue

        selected[f"S{i + 1}"] = window

        seen.add(window)

        if len(selected) == count:
            break

    return selected


# ============================================================
# PREPARE NUMBERED EVIDENCE
# ============================================================

def _source_context(
    chunks: list[dict],
) -> tuple[str, dict]:

    sections = []
    source_map = {}

    for index, chunk in enumerate(
        chunks,
        1,
    ):

        eid = f"E{index}"

        spans = _windows(
            str(chunk.get("text", ""))
        )

        source_map[eid] = spans

        section = [
            f"[{eid}] "
            f"Speaker: {chunk.get('speaker', 'Unknown')} | "
            f"Date: {chunk.get('date', '')} | "
            f"Agenda: {chunk.get('agenda', '')}"
        ]

        section.extend(
            f"[{sid}] {span}"
            for sid, span in spans.items()
        )

        sections.append(
            "\n".join(section)
        )

    return (
        "\n\n".join(sections),
        source_map,
    )


# ============================================================
# INDEPENDENT NLI VERIFIER
# ============================================================

@lru_cache(maxsize=1)
def _verifier():

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        NLI_MODEL,
        max_length=384,
    )

    # Read the model's configured label order.
    labels = {
        int(i): str(value).lower()
        for i, value
        in model.model.config.id2label.items()
    }

    expected = {
        "entailment",
        "contradiction",
        "neutral",
    }

    if set(labels.values()) != expected:
        raise ValueError(
            f"Unexpected NLI labels: {labels}"
        )

    return model, labels


# ============================================================
# STRUCTURED CLAIM GENERATION
# ============================================================

def _generate_claims(
    engine,
    model_name: str,
    question: str,
    context: str,
) -> list[dict]:

    prompt = f"""
You are examining UK parliamentary speeches.

Generate atomic, source-grounded claims.

Return only a JSON object:

{{
  "claims": [
    {{
      "evidence_id": "E1",
      "span_id": "S1",
      "claim": "One short sentence naming the speaker."
    }}
  ]
}}

RULES:

1. One simple factual clause per claim.
2. The claim must name the correct speaker.
3. A speaker's personal position must be explicit
   in the supporting source window.
4. When reporting a petition or another person's
   argument, say that the speaker REPORTED it.
5. Do not confuse supporting Brexit with supporting
   another Brexit referendum.
6. Do not infer party-wide or parliamentary consensus.
7. Use only existing E and S identifiers.
8. Use at most one claim per E identifier.
9. Produce no more than 12 claims.
10. Prefer accurate descriptions to guessed stances.
11. Each claim must be at most 175 characters.
12. Do not include citation tags in claim text.

RESEARCH QUESTION:
{question}

NUMBERED SOURCE WINDOWS:
{context}
"""

    response = (
        engine.llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0.0,
            max_tokens=1500,
        )
    )

    raw = (
        response.choices[0].message.content
        or ""
    ).strip()

    if raw.startswith("```"):

        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
        ).strip()

    parsed = json.loads(raw)

    if (
        not isinstance(parsed, dict)
        or not isinstance(
            parsed.get("claims"),
            list,
        )
    ):
        raise ValueError(
            "Synthesis model did not return "
            "a JSON claims array."
        )

    return parsed["claims"]


# ============================================================
# VERIFY CLAIMS USING NLI
# ============================================================

def _verify_claims(
    claims,
    chunks,
    source_map,
    verifier,
    labels,
):

    accepted = []

    used = set()

    for item in claims[:24]:

        if not isinstance(item, dict):
            continue

        eid = item.get(
            "evidence_id"
        )

        sid = item.get(
            "span_id"
        )

        claim = item.get(
            "claim"
        )

        if not isinstance(eid, str):
            continue

        if not re.fullmatch(
            r"E[1-9]\d*",
            eid,
        ):
            continue

        index = int(
            eid[1:]
        ) - 1

        if index >= len(chunks):
            continue

        if eid in used:
            continue

        speaker = str(
            chunks[index].get(
                "speaker", ""
            )
        )

        if not isinstance(sid, str):
            continue

        if sid not in source_map.get(eid, {}):
            continue

        if not isinstance(claim, str):
            continue

        if not 18 <= len(claim) <= 190:
            continue

        if not speaker:
            continue

        if speaker.casefold() not in claim.casefold():
            continue

        if re.search(
            r"\[E\d+\]",
            claim,
        ):
            continue

        # Allow other selected windows from the
        # same speech to provide additional context.
        windows = [
            source_map[eid][sid]
        ]

        windows += [
            value
            for key, value in source_map[eid].items()
            if key != sid
        ]

        best = None

        for premise in windows:

            original_text = str(
                chunks[index].get("text", "")
            )

            if premise not in original_text:
                continue

            result = verifier.predict(
                [
                    (
                        f"Parliamentary speech by "
                        f"{speaker}: {premise}",
                        claim,
                    )
                ],
                apply_softmax=True,
            )[0]

            probabilities = {
                labels[i]: float(result[i])
                for i in labels
            }

            if (
                best is None
                or probabilities["entailment"]
                > best["entailment"]
            ):
                best = probabilities

        if best is None:
            continue

        logger.warning(
            "NLI %s: entailment=%.3f "
            "contradiction=%.3f neutral=%.3f | %s",
            eid,
            best["entailment"],
            best["contradiction"],
            best["neutral"],
            claim,
        )

        if (
            best["entailment"] < MIN_ENTAILMENT
            or best["contradiction"]
            > MAX_CONTRADICTION
        ):
            continue

        accepted.append(
            (
                index,
                claim.strip().rstrip(". "),
                best["entailment"],
            )
        )

        used.add(eid)

    return accepted


# ============================================================
# ORIGINAL-SOURCE FALLBACK
# ============================================================

def _quotation(
    index: int,
    chunk: dict,
) -> str:

    return (
        f"{chunk.get('speaker', 'Unknown')} "
        f"({chunk.get('date', '')}), "
        f"speech excerpt: "
        f"“{_quote(chunk)}” "
        f"[E{index + 1}]."
    )


# ============================================================
# CHRONOLOGICAL ANSWER ASSEMBLY
# ============================================================

def _assemble(
    chunks: list[dict],
    accepted: list[tuple],
    question: str = "",
) -> str:

    """
    Build the final response deterministically.

    Only NLI-accepted claims are paraphrased.
    Remaining content uses original quotations.

    No second free-form LLM generation call.
    """

    years = defaultdict(list)

    for index, chunk in enumerate(chunks):

        year = str(
            chunk.get("date", "")
        )[:4]

        years[year].append(index)

    accepted_by_year = defaultdict(list)

    for index, claim, confidence in accepted:

        year = str(
            chunks[index].get("date", "")
        )[:4]

        accepted_by_year[year].append(
            (
                index,
                claim,
                confidence,
            )
        )

    year_keys = sorted(years)

    if not year_keys:
        return "No dated evidence was retrieved."

    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    opening = (
        f"From {year_keys[0]} to {year_keys[-1]}, "
        "the selected speeches illustrate arguments "
        "and proposals discussed in Parliament."
    )

    # Add a limited comparison when it can be
    # supported by actual source passages.
    #
    # This does not measure changes in support.
    if re.search(
        r"second.*referendum|people['’]?s vote",
        question,
        re.I,
    ):

        early = next(
            (
                i
                for i, chunk in enumerate(chunks)
                if str(
                    chunk.get("date", "")
                )[:4] in {"2016", "2017"}
                and re.search(
                    r"another referendum|"
                    r"second referendum|"
                    r"further referendum",
                    str(chunk.get("text", "")),
                    re.I,
                )
            ),
            None,
        )

        later = next(
            (
                i
                for i, chunk in enumerate(chunks)
                if str(
                    chunk.get("date", "")
                )[:4] in {"2018", "2019"}
                and re.search(
                    r"public vote|"
                    r"confirmatory ballot|"
                    r"vote on the "
                    r"(?:brexit )?deal",
                    str(chunk.get("text", "")),
                    re.I,
                )
            ),
            None,
        )

        if early is not None and later is not None:

            opening += (
                " Earlier entries raise another vote "
                "following the 2016 result "
                f"[E{early + 1}], while later entries "
                "include proposals for a public vote "
                "on the negotiated deal "
                f"[E{later + 1}]."
            )

    paragraphs = [
        opening
        + " These examples do not measure "
        "parliamentary consensus."
    ]

    # --------------------------------------------------------
    # YEAR-BY-YEAR SYNTHESIS
    # --------------------------------------------------------

    for position, year in enumerate(year_keys):

        chosen = sorted(
            accepted_by_year[year],
            key=lambda row: (
                -row[2],
                row[0],
            ),
        )[:2]

        statements = [
            f"{claim} [E{index + 1}]."
            for index, claim, _
            in chosen
        ]

        used_indices = {
            index
            for index, _, _
            in chosen
        }

        # Fill missing space with source quotations.
        if len(statements) < 2:

            backups = [
                index
                for index in years[year]
                if index not in used_indices
            ]

            backups.sort(
                key=lambda index: float(
                    chunks[index].get(
                        "rerank_score"
                    ) or 0
                ),
                reverse=True,
            )

            needed = 2 - len(statements)

            for index in backups[:needed]:

                statements.append(
                    _quotation(
                        index,
                        chunks[index],
                    )
                )

        # Small narrative transitions.
        lead = (
            "In"
            if position != 1
            else "By"
        )

        paragraphs.append(
            f"{lead} **{year}**, "
            + " ".join(statements)
        )

    # --------------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------------

    paragraphs.append(
        "Together, these dated examples show "
        "the arguments appearing in the "
        "retrieved record. They do not establish "
        "whether Parliament as a whole became "
        "more or less supportive over time."
    )

    return "\n\n".join(paragraphs)


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def generate_grounded_answer(
    engine,
    question: str,
    final_chunks: list[dict],
) -> str:

    """
    Drop-in replacement for the function already
    called by app.graphrag_service.py.
    """

    if not final_chunks:
        return (
            "No relevant parliamentary "
            "speeches were retrieved."
        )

    chunks = sorted(
        final_chunks,
        key=lambda item: (
            str(item.get("date", "")),
            str(item.get("speaker", "")),
        ),
    )

    model_name = os.getenv(
        "GRAPHRAG_SYNTHESIS_MODEL",
        "",
    ).strip()

    if not model_name:

        logger.warning(
            "Synthesis model unset: "
            "quotation fallback only."
        )

        return _assemble(
            chunks,
            [],
            question,
        )

    try:

        # ----------------------------------------
        # GENERATE CANDIDATE CLAIMS
        # ----------------------------------------

        context, sources = _source_context(
            chunks
        )

        candidates = _generate_claims(
            engine,
            model_name,
            question,
            context,
        )

        logger.warning(
            "Hybrid generated %d claims.",
            len(candidates),
        )

        # ----------------------------------------
        # VERIFY CANDIDATE CLAIMS
        # ----------------------------------------

        verifier, labels = _verifier()

        accepted = _verify_claims(
            candidates,
            chunks,
            sources,
            verifier,
            labels,
        )

        logger.warning(
            "Hybrid accepted %d/%d claims.",
            len(accepted),
            len(candidates),
        )

        # ----------------------------------------
        # BUILD FINAL CHRONOLOGICAL ANSWER
        # ----------------------------------------

        return _assemble(
            chunks,
            accepted,
            question,
        )

    except Exception:

        logger.exception(
            "Hybrid generation failed; "
            "returning source quotations."
        )

        return _assemble(
            chunks,
            [],
            question,
        )
