
"""Evidence-first temporal answer generation for Brexit GraphRAG.

Place this file at app/temporal_grounded_v2.py.

No Ollama or NLI calls are made here. This generator uses
only the evidence already selected by the retrieval layer.
Original [E1], [E2], ... citation numbering is preserved.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()


# ============================================================
# QUESTION AND SUBJECT DETECTION
# ============================================================

_RANGE = re.compile(
    r"\b(?:between|from)\s+(20\d{2})\s+"
    r"(?:and|to|through|-)\s+(20\d{2})\b",
    re.I,
)

_REFERENDUM = re.compile(
    r"\b(?:second|another|further|repeat|new)\s+"
    r"(?:brexit\s+|eu\s+)?referendum\b"
    r"|\b(?:people['’]s|public|confirmatory)\s+"
    r"(?:vote|ballot)\b"
    r"|\breferendum\s+on\s+(?:the|a|our)\s+"
    r"(?:deal|terms|agreement|exit|package)\b"
    r"|\b(?:final|decision)\s+referendum\b",
    re.I,
)

_CUSTOMS = re.compile(
    r"\bcustoms union\b",
    re.I,
)


# ============================================================
# EVIDENCE QUALITY CUES
# ============================================================

# These are ranking hints only.
# They are NOT verified political stance classifications.

_REF_SUPPORT = re.compile(
    r"\b(?:we|i|labour|the party)\s+"
    r"(?:would\s+|will\s+|do\s+)?support\s+"
    r"(?:a|the|another|second)\s+"
    r"(?:public\s+vote|referendum|confirmatory\s+ballot)\b"
    r"|\b(?:should|must|need\s+to)\s+"
    r"(?:have|hold|offer)\s+"
    r"(?:a|another|second)\s+"
    r"(?:referendum|public vote)\b"
    r"|\b(?:our proposal|i propose|we propose)\b"
    r".{0,140}\b"
    r"(?:referendum|confirmatory ballot|public vote)\b",
    re.I | re.S,
)

_REF_OPPOSE = re.compile(
    r"\b(?:we|i|government)\s+"
    r"(?:will|would|do|does|shall)\s+not\s+"
    r"(?:have|hold|support|accept)\s+"
    r"(?:a|another|second)\s+"
    r"(?:referendum|public vote)\b"
    r"|\b(?:no|not a|against a|oppose a)\s+"
    r"second\s+referendum\b"
    r"|\bsecond\s+referendum\b.{0,95}"
    r"\b(?:wrong|dangerous|undemocratic|"
    r"uncertainty|betrayal)\b",
    re.I | re.S,
)

_CUSTOMS_RETAIN = re.compile(
    r"\b(?:permanent|comprehensive|continued|new)"
    r"\s+customs union\b"
    r"|\b(?:remain|stay|membership|keep|maintain)\b"
    r".{0,70}\bcustoms union\b"
    r"|\bcustoms union\b.{0,70}"
    r"\b(?:jobs|frictionless|trade barriers|hard border)\b",
    re.I | re.S,
)

_CUSTOMS_LEAVE = re.compile(
    r"\b(?:leave|leaving|outside|exit|out of)\b"
    r".{0,70}\bcustoms union\b"
    r"|\bcustoms union\b.{0,100}"
    r"\b(?:independent trade policy|trade deals|own trade)\b",
    re.I | re.S,
)

_ARGUMENT = re.compile(
    r"\b(?:trade deals|rules of origin|frictionless|"
    r"border|independent trade|tariff|jobs|workers|"
    r"sovereignty|customs arrangements|market access|"
    r"withdrawal agreement|negotiat\w*)\b",
    re.I,
)

_LOW_VALUE = re.compile(
    r"^(?:i am (?:extremely )?grateful|i thank|"
    r"the focus of today['’]s debate|"
    r"let me now deal|i welcome the opportunity)\b",
    re.I,
)

_REPORTED = re.compile(
    r"\b(?:opponents? might|some have said|"
    r"he said|she said|they argue|"
    r"the prime minister described|"
    r"those who argue|critics say)\b",
    re.I,
)

_SCOTLAND = re.compile(
    r"\b(?:indyref2|scottish independence|"
    r"scotland['’]s independence|"
    r"independence referendum|"
    r"claim of right for scotland)\b",
    re.I,
)


# ============================================================
# TEXT SEGMENTATION
# ============================================================

def _sentences(text: str) -> list[str]:
    """
    Split speech text without breaking after
    common parliamentary honorifics.
    """

    protected = re.sub(
        r"\b(?:hon|Mr|Mrs|Ms|Dr|Prof|Rt|St)\.",
        lambda match: (
            match.group(0)[:-1] + "<DOT>"
        ),
        text,
        flags=re.I,
    )

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z“‘])",
        protected,
    )

    return [
        part.replace("<DOT>", ".").strip()
        for part in parts
        if part.strip()
    ]


# ============================================================
# QUESTION INTENT
# ============================================================

def _topic(question: str) -> str:

    q = question.casefold()

    if "customs union" in q:
        return "customs"

    if (
        "referendum" in q
        or "public vote" in q
        or "confirmatory" in q
    ):
        return "referendum"

    return "general"


# ============================================================
# TOPIC DISAMBIGUATION
# ============================================================

def _wrong_topic(
    chunk: dict,
    topic: str,
) -> bool:
    """
    Exclude obvious Scottish independence
    referendum results from Brexit queries.
    """

    if topic != "referendum":
        return False

    agenda = str(
        chunk.get("agenda", "")
    )

    text = str(
        chunk.get("text", "")
    )

    return bool(
        "claim of right for scotland"
        in agenda.casefold()

        or (
            _SCOTLAND.search(text)
            and not re.search(
                r"\b(?:brexit|european union|"
                r"eu referendum|withdrawal agreement)\b",
                text,
                re.I,
            )
        )
    )


# ============================================================
# SENTENCE RELEVANCE
# ============================================================

def _relevance(
    sentence: str,
    topic: str,
) -> float:
    """
    Score the actual source sentence,
    not the retrieval label of its parent chunk.
    """

    s = sentence.strip()

    if len(s) < 48:
        return -100.0

    # --------------------------------------------------------
    # REFERENDUM QUESTIONS
    # --------------------------------------------------------

    if topic == "referendum":

        if not _REFERENDUM.search(s):
            return -100.0

        score = 11.0

        if _REF_SUPPORT.search(s):
            score += 6

        if _REF_OPPOSE.search(s):
            score += 6

        if re.search(
            r"\b(?:deal|withdrawal|negotiat\w*|"
            r"public|result|mandate|democra\w*)\b",
            s,
            re.I,
        ):
            score += 2

        if _REPORTED.search(s):
            score -= 6

    # --------------------------------------------------------
    # CUSTOMS UNION QUESTIONS
    # --------------------------------------------------------

    elif topic == "customs":

        if not _CUSTOMS.search(s):
            return -100.0

        score = 11.0

        if (
            _CUSTOMS_RETAIN.search(s)
            or _CUSTOMS_LEAVE.search(s)
        ):
            score += 7

        if _ARGUMENT.search(s):
            score += 3

        if _REPORTED.search(s):
            score -= 3

    # --------------------------------------------------------
    # OTHER TEMPORAL QUESTIONS
    # --------------------------------------------------------

    else:

        score = 2.0

        if _ARGUMENT.search(s):
            score += 3

    # --------------------------------------------------------
    # GENERAL QUALITY PENALTIES
    # --------------------------------------------------------

    if _LOW_VALUE.search(s):
        score -= 12

    if len(s) > 500:
        score -= 4

    if "€" in s:
        score -= 2

    if "?" in s:
        score -= 1.5

    return score


# ============================================================
# SELECT A SOURCE EXCERPT
# ============================================================

def _best_excerpt(
    chunk: dict,
    topic: str,
) -> tuple[float, str]:
    """
    Select a directly relevant quotation.

    Do not invent missing context or generate
    a paraphrase of an ambiguous sentence.
    """

    parts = _sentences(
        str(
            chunk.get("text", "")
        )
    )

    scored = [
        (
            _relevance(sentence, topic),
            index,
            sentence,
        )

        for index, sentence
        in enumerate(parts)
    ]

    if not scored:
        return -100.0, ""

    score, index, excerpt = max(
        scored,
        key=lambda item: (
            item[0],
            -item[1],
        ),
    )

    threshold = (
        10.8
        if topic != "general"
        else 3.0
    )

    if score < threshold:
        return score, ""

    # Restore antecedents for short statements
    # beginning "That", "This", or "It".

    if index > 0 and re.match(
        r"^(?:that|this|it|those|therefore)\b",
        excerpt,
        re.I,
    ):

        previous = parts[index - 1]

        if (
            len(previous)
            + len(excerpt)
            < 520
        ):
            excerpt = (
                previous
                + " "
                + excerpt
            )

    return score, excerpt


# ============================================================
# HEURISTIC STANCE LABEL
# ============================================================

def _stance(
    excerpt: str,
    topic: str,
) -> str:
    """
    Used for evidence selection diagnostics only.
    Never presented as verified political stance.
    """

    if topic == "referendum":

        if _REPORTED.search(excerpt):
            return "descriptive"

        positive = bool(
            _REF_SUPPORT.search(excerpt)
        )

        negative = bool(
            _REF_OPPOSE.search(excerpt)
        )

        if positive and not negative:
            return "support"

        if negative and not positive:
            return "oppose"

    elif topic == "customs":

        retain = bool(
            _CUSTOMS_RETAIN.search(excerpt)
        )

        leave = bool(
            _CUSTOMS_LEAVE.search(excerpt)
        )

        if retain and not leave:
            return "retain"

        if leave and not retain:
            return "leave"

    return "discussed"


# ============================================================
# REQUESTED YEARS
# ============================================================

def _requested_years(
    question: str,
    chunks: list[dict],
) -> list[str]:

    match = _RANGE.search(question)

    if match:

        start = int(match.group(1))
        end = int(match.group(2))

        if 2015 <= start <= end <= 2019:

            return [
                str(year)
                for year in range(
                    start,
                    end + 1,
                )
            ]

    return sorted(
        {
            str(chunk.get("date", ""))[:4]

            for chunk in chunks

            if re.match(
                r"^201[5-9]",
                str(chunk.get("date", "")),
            )
        }
    )


# ============================================================
# YEAR-LEVEL EVIDENCE SELECTION
# ============================================================

def _choose_year(
    rows: list[dict],
    cross_party: bool,
) -> list[dict]:
    """
    Select up to two substantive statements per year.

    Prefer different speakers and, for cross-party
    questions, different parties where available.
    """

    if not rows:
        return []

    ranked = sorted(
        rows,
        key=lambda item: (
            -item["score"],
            item["index"],
        ),
    )

    selected = [
        ranked[0]
    ]

    alternatives = [
        row
        for row in ranked[1:]

        if row["speaker"].casefold()
        != selected[0]["speaker"].casefold()
    ]

    if not alternatives:
        return selected

    if cross_party:

        other_party = [
            row
            for row in alternatives

            if row["party"]
            != selected[0]["party"]
        ]

        if other_party:
            alternatives = other_party

    different = [
        row
        for row in alternatives

        if (
            row["stance"]
            != selected[0]["stance"]

            and row["stance"]
            not in {
                "discussed",
                "descriptive",
            }
        )
    ]

    if (
        different
        and different[0]["score"]
        >= alternatives[0]["score"] - 4
    ):

        selected.append(
            different[0]
        )

    else:

        selected.append(
            alternatives[0]
        )

    return selected


# ============================================================
# DIAGNOSTIC TRACE
# ============================================================

def _trace(
    question: str,
    topic: str,
    rows: list[dict],
    picked: list[dict],
    years: list[str],
) -> None:

    destination = os.getenv(
        "GRAPHRAG_TEMPORAL_TRACE",
        "data/temporal_generation_trace.jsonl",
    ).strip()

    if destination.casefold() in {
        "",
        "off",
        "false",
        "0",
    }:
        return

    path = Path(
        destination
    )

    if not path.is_absolute():
        path = _ROOT / path

    payload = {
        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "question": question,
        "topic": topic,
        "requested_years": years,

        "candidate_counts": dict(
            Counter(
                row["year"]
                for row in rows
            )
        ),

        "selected_ids": [
            f"E{row['index'] + 1}"
            for row in picked
        ],

        "selected_stances": {
            f"E{row['index'] + 1}": row["stance"]
            for row in picked
        },

        "mode": "source_quotation_only",
    }

    try:

        with _LOCK:

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with path.open(
                "a",
                encoding="utf-8",
            ) as handle:

                handle.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    except OSError:

        log.exception(
            "Could not write temporal generation trace."
        )


# ============================================================
# PUBLIC TEMPORAL GENERATOR
# ============================================================

def generate_grounded_answer(
    engine,
    question: str,
    final_chunks: list[dict],
) -> str:
    """
    Build a conservative chronological answer.

    Compatible with the existing service call.

    The engine parameter is preserved but not used.
    No LLM or NLI is loaded by this function.

    [E1] is final_chunks[0], etc.
    """

    del engine

    chunks = list(
        final_chunks or []
    )

    if not chunks:

        return (
            "I could not find sufficiently relevant "
            "parliamentary speech evidence to answer "
            "this question."
        )

    topic = _topic(question)

    years = _requested_years(
        question,
        chunks,
    )

    cross_party = bool(
        re.search(
            r"\bcross[- ]party\b",
            question,
            re.I,
        )
    )

    candidates = []

    # --------------------------------------------------------
    # BUILD SUBSTANTIVE EVIDENCE CANDIDATES
    # --------------------------------------------------------

    for index, chunk in enumerate(chunks):

        date = str(
            chunk.get("date", "")
        )

        year = date[:4]

        if (
            year not in years
            or _wrong_topic(
                chunk,
                topic,
            )
        ):
            continue

        if (
            not chunk.get("speaker")
            or not chunk.get("text")
        ):
            continue

        score, quote = _best_excerpt(
            chunk,
            topic,
        )

        if not quote:
            continue

        score += (
            0.10
            * float(
                chunk.get(
                    "rerank_score"
                ) or 0
            )
        )

        candidates.append(
            {
                "index": index,
                "year": year,
                "date": date,

                "speaker": str(
                    chunk.get("speaker")
                ),

                "party": str(
                    chunk.get("party")
                    or "unrecorded party"
                ),

                "quote": quote,
                "score": score,

                "stance": _stance(
                    quote,
                    topic,
                ),
            }
        )

    # --------------------------------------------------------
    # SELECT EVIDENCE FOR EACH YEAR
    # --------------------------------------------------------

    by_year = {
        year: _choose_year(
            [
                row
                for row in candidates
                if row["year"] == year
            ],
            cross_party,
        )
        for year in years
    }

    selected = [
        row
        for year in years
        for row in by_year[year]
    ]

    _trace(
        question,
        topic,
        candidates,
        selected,
        years,
    )

    if not selected:

        return (
            "The selected parliamentary speeches "
            "do not contain sufficiently substantive, "
            "on-topic passages for a reliable "
            "year-by-year answer."
        )

    paragraphs = []

    # --------------------------------------------------------
    # LIMITED, SOURCE-BASED INTRODUCTION
    # --------------------------------------------------------

    if topic == "referendum":

        early = next(
            (
                row
                for row in by_year.get(
                    "2016", []
                )

                if _REFERENDUM.search(
                    row["quote"]
                )
            ),
            None,
        )

        later = next(
            (
                row
                for row in by_year.get(
                    "2019", []
                )

                if (
                    "confirmatory ballot"
                    in row["quote"].casefold()

                    and re.search(
                        r"\b(?:our proposal|we propose|"
                        r"we would|labour would)\b",
                        row["quote"],
                        re.I,
                    )
                )
            ),
            None,
        )

        if early and later:

            paragraphs.append(
                "The retrieved speeches show that "
                "a further Brexit vote was already "
                "being debated in 2016 "
                f"[E{early['index'] + 1}], while "
                "a selected 2019 speech set out "
                "a confirmatory-ballot proposal "
                f"[E{later['index'] + 1}]. "
                "This is not evidence of how much "
                "parliamentary or cross-party support changed."
            )

        else:

            paragraphs.append(
                "The following passages document "
                "the referendum debate at different dates. "
                "They do not measure the level of "
                "parliamentary or cross-party support."
            )

    elif topic == "customs":

        paragraphs.append(
            "The retrieved speeches provide dated "
            "examples of the customs-union debate. "
            "They are not, by themselves, evidence "
            "of how many MPs supported each option."
        )

    else:

        paragraphs.append(
            "The following is a chronology of "
            "statements found in the selected "
            "parliamentary speeches."
        )

    # --------------------------------------------------------
    # GENERATE THE CHRONOLOGY
    # --------------------------------------------------------

    for year in years:

        picks = by_year[year]

        if not picks:

            paragraphs.append(
                f"**{year}:** "
                "No sufficiently substantive passage "
                "was selected for this year; "
                "the available evidence is insufficient "
                "to characterise its debate."
            )

            continue

        statements = []

        for item in picks:

            statements.append(
                f"{item['speaker']} "
                f"({item['party']}, {item['date']}) "
                f"said: “{item['quote']}” "
                f"[E{item['index'] + 1}]."
            )

        paragraphs.append(
            f"**{year}:** "
            + " ".join(statements)
        )

    # --------------------------------------------------------
    # EXPLICIT INTERPRETATION LIMIT
    # --------------------------------------------------------

    if topic == "customs":

        paragraphs.append(
            "**Interpretation:** These quotations "
            "identify issues raised in the selected "
            "years; they do not alone justify a claim "
            "about an overall shift in Parliament's position."
        )

    elif cross_party:

        paragraphs.append(
            "**Interpretation:** Party labels identify "
            "the speakers shown, but this sample does "
            "not measure changes in cross-party support."
        )

    else:

        paragraphs.append(
            "**Interpretation:** The examples illustrate "
            "arguments over time, not a measured change "
            "in parliamentary consensus."
        )

    return "\n\n".join(
        paragraphs
    )
