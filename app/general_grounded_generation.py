
"""General source-grounded answers for the FastAPI app.

Reuses the temporal generator's cached NLI verifier.
Does not change the original GraphRAG engine.

Citation IDs E1, E2, etc. follow final_chunks order,
which must match the API's evidence array.
"""

from __future__ import annotations

import json
import logging
import os
import re

from collections import Counter

from app.grounded_generation import (
    _verifier,
    _verify_claims,
)


logger = logging.getLogger(__name__)

BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
    r"|(?<=[.!?])(?=[A-Z])"
)

WORDS = re.compile(
    r"[a-z0-9]+(?:'[a-z]+)?",
    re.I,
)

STOP = {
    "what", "which", "where", "when",
    "why", "how", "was", "were",
    "is", "are", "the", "and",
    "for", "from", "their", "with",
    "between", "did", "does", "do",
    "about", "main", "on", "of",
    "to", "in", "a", "an",
    "it", "its", "that", "this",
}


# ============================================================
# GENERAL TEXT PROCESSING
# ============================================================

def _sentences(text: str) -> list[str]:

    source = str(text or "")

    return [
        part.strip()
        for part in BOUNDARY.split(source)
        if part.strip()
        and part.strip() in source
    ]


def _terms(text: str) -> set[str]:

    return {
        word.casefold()
        for word in WORDS.findall(text)
        if len(word) > 2
        and word.casefold() not in STOP
    }


# ============================================================
# SOURCE WINDOWS
# ============================================================

def _source_windows(
    chunks: list[dict],
    question: str,
) -> tuple[str, dict]:

    """
    Select short, exact passages for any
    non-temporal research question.
    """

    query_terms = _terms(question)

    source_map = {}
    sections = []

    for index, chunk in enumerate(
        chunks,
        1,
    ):

        eid = f"E{index}"

        original = str(
            chunk.get("text") or ""
        )

        sentences = _sentences(original)

        ranked = sorted(
            range(len(sentences)),
            key=lambda i: (
                -len(
                    _terms(sentences[i])
                    & query_terms
                ),
                i,
            ),
        )

        windows = {}
        seen = set()

        for i in ranked:

            proposed = " ".join(
                sentences[
                    max(0, i - 1):i + 2
                ]
            )

            if (
                proposed not in original
                or len(proposed) > 850
            ):
                proposed = sentences[i]

            if len(proposed) > 850:
                proposed = (
                    proposed[:850]
                    .rsplit(" ", 1)[0]
                )

            if (
                proposed
                and proposed in original
                and proposed not in seen
            ):

                windows[
                    f"S{i + 1}"
                ] = proposed

                seen.add(proposed)

            if len(windows) >= 4:
                break

        source_map[eid] = windows

        header = (
            f"[{eid}] "
            f"{chunk.get('speaker', 'Unknown')} | "
            f"{chunk.get('party', '')} | "
            f"{chunk.get('date', '')} | "
            f"{chunk.get('agenda', '')}"
        )

        sections.append(
            "\n".join(
                [header]
                + [
                    f"[{sid}] {window}"
                    for sid, window
                    in windows.items()
                ]
            )
        )

    return (
        "\n\n".join(sections),
        source_map,
    )


# ============================================================
# STRUCTURED CLAIM GENERATION
# ============================================================

def _request_claims(
    engine,
    question: str,
    context: str,
    model_name: str,
) -> list[dict]:

    prompt = f"""
Use only the numbered UK parliamentary excerpts
below to answer the question.

Return ONLY a JSON object with a "claims" array.

Each item must contain:
- evidence_id
- span_id
- claim

RULES:

1. One factual, independently checkable
   statement per claim.

2. Name the speaker exactly as shown.

3. Prefer statements that directly
   address the research question.

4. Do not attribute a quotation, petition,
   objection or reported opinion to a speaker
   as their own position without explicit evidence.

5. If describing a party position, say that
   the speaker reported the party position.

6. Do not merge arguments from different speakers.

7. Do not invent legal definitions, dates,
   events, motives, agreements or trends.

8. Use only existing E and S identifiers.

9. Never include citation tags inside a claim.

10. At most eight claims, each no longer
    than 160 characters.

11. Omit claims whose support is unclear.

OUTPUT EXAMPLE:

{{
  "claims": [
    {{
      "evidence_id": "E1",
      "span_id": "S2",
      "claim": "Example Speaker argued that the proposal would create uncertainty."
    }}
  ]
}}

QUESTION:
{question}

NUMBERED EVIDENCE:
{context}
"""

    result = (
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
            max_tokens=1400,
        )
    )

    raw = (
        result.choices[0]
        .message.content or ""
    ).strip()

    raw = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw,
    ).strip()

    data = json.loads(raw)

    if (
        not isinstance(data, dict)
        or not isinstance(
            data.get("claims"),
            list,
        )
    ):
        raise ValueError(
            "Expected a JSON object "
            "with a claims array."
        )

    return data["claims"]


# ============================================================
# SOURCE QUOTATION FALLBACK
# ============================================================

def _quote(
    chunk: dict,
    question: str,
    limit: int = 290,
) -> str:

    source = str(
        chunk.get("text") or ""
    )

    sentences = _sentences(source)

    if not sentences:
        return ""

    terms = _terms(question)

    def relevance(sentence: str) -> tuple:

        tokens = _terms(sentence)

        overlap = sum(
            3 if word in {
                "backstop",
                "referendum",
                "brexit",
                "article",
                "customs",
                "union",
                "border",
            }
            else 1
            for word in (tokens & terms)
        )

        substantive = bool(
            re.search(
                r"\b(?:would|should|must|"
                r"because|concern|risk|oppose|"
                r"support|argue|protect|ensure|"
                r"prevent)\w*\b",
                sentence,
                re.I,
            )
        )

        return (
            overlap + 2 * substantive,
            min(len(sentence), 230),
        )

    best = max(
        sentences,
        key=relevance,
    )

    if len(best) <= limit:
        return best

    fragment = best[:limit]

    if " " in fragment:
        fragment = fragment.rsplit(
            " ",
            1,
        )[0]

    return fragment + "…"


# ============================================================
# ASSEMBLE GENERAL ANSWER
# ============================================================

def _assemble(
    chunks: list[dict],
    question: str,
    accepted: list[tuple],
) -> str:

    if not chunks:
        return (
            "No relevant parliamentary "
            "speeches were retrieved."
        )

    distinct = []
    speaker_counts = Counter()

    for index, claim, confidence in sorted(
        accepted,
        key=lambda item: -item[2],
    ):

        speaker = str(
            chunks[index].get("speaker", "")
        )

        if speaker_counts[speaker] >= 2:
            continue

        speaker_counts[speaker] += 1

        distinct.append(
            (index, claim)
        )

        if len(distinct) >= 5:
            break

    distinct.sort(
        key=lambda item: item[0]
    )

    sections = []

    if distinct:

        sections.append(
            "**According to the retrieved "
            "parliamentary speeches:**"
        )

        sections.append(
            " ".join(
                f"{claim.rstrip('. ')} "
                f"[E{index + 1}]."
                for index, claim
                in distinct
            )
        )

    # If too few claims pass NLI, include
    # original excerpts with exact citations.
    if len(distinct) < 3:

        used = {
            index
            for index, _ in distinct
        }

        backup = sorted(
            (
                i
                for i in range(len(chunks))
                if i not in used
            ),
            key=lambda i: float(
                chunks[i].get(
                    "rerank_score"
                ) or 0
            ),
            reverse=True,
        )

        selections = []

        needed = max(
            1,
            3 - len(distinct),
        )

        for i in backup[:needed]:

            quote = _quote(
                chunks[i],
                question,
            )

            if not quote:
                continue

            selections.append(
                f"{chunks[i].get('speaker', 'Unknown')}: "
                f"“{quote}” [E{i + 1}]"
            )

        if selections:

            sections.append(
                "**Additional source excerpts:** "
                + " ".join(selections)
            )

    if not sections:
        return (
            "The selected speeches do not "
            "provide enough evidence to "
            "answer reliably."
        )

    if len(distinct) < 2:

        sections.append(
            "The retrieved evidence is "
            "insufficient for a full "
            "answer to this question."
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# MAIN GENERAL GROUNDED GENERATOR
# ============================================================

def generate_general_grounded_answer(
    engine,
    question: str,
    final_chunks: list[dict],
) -> str:

    """
    Produce a cited non-temporal answer.

    Evidence numbering must follow the
    original final_chunks order.
    """

    chunks = list(
        final_chunks or []
    )

    if not chunks:
        return (
            "No relevant parliamentary "
            "speeches were retrieved."
        )

    model_name = os.getenv(
        "GRAPHRAG_SYNTHESIS_MODEL",
        "",
    ).strip()

    if not model_name:

        logger.warning(
            "General synthesis model unset; "
            "using source quotations."
        )

        return _assemble(
            chunks,
            question,
            [],
        )

    try:

        # Prepare question-specific source windows.
        context, source_map = _source_windows(
            chunks,
            question,
        )

        # Generate source-linked claims.
        candidates = _request_claims(
            engine,
            question,
            context,
            model_name,
        )

        # Reuse the existing cached NLI verifier.
        verifier, labels = _verifier()

        accepted = _verify_claims(
            candidates,
            chunks,
            source_map,
            verifier,
            labels,
        )

        logger.warning(
            "General synthesis accepted %s/%s claims.",
            len(accepted),
            len(candidates),
        )

        return _assemble(
            chunks,
            question,
            accepted,
        )

    except Exception:

        logger.exception(
            "General synthesis failed; "
            "using cited source quotations."
        )

        return _assemble(
            chunks,
            question,
            [],
        )
