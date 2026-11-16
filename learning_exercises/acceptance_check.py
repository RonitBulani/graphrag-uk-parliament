
"""Final GraphRAG API acceptance checks.

Usage:
    python learning_exercises/acceptance_check.py --saved data/latest_response.json

    python learning_exercises/acceptance_check.py --live --limit 8

    python learning_exercises/acceptance_check.py --live --all

Requires only Python's standard library.

Automated checks cover API structure and known regression
patterns. They do not prove historical factual correctness.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

from collections import Counter
from pathlib import Path


CITATION = re.compile(
    r"\[E([1-9]\d*)\]"
)

YEAR_RANGE = re.compile(
    r"\bbetween\s+(20\d{2})\s+and\s+(20\d{2})\b",
    re.I,
)

SECOND_VOTE = re.compile(
    r"\bsecond\s+(?:Brexit\s+)?referendum\b"
    r"|\bpeople['’]s vote\b",
    re.I,
)

REFERENDUM_QUESTION = (
    "How did parliamentary discussions about "
    "a second Brexit referendum evolve "
    "between 2016 and 2019?"
)


# Known errors encountered in previous runs.
#
# These expressions are regression warnings,
# not political stance classifiers.

KNOWN_ERRORS = {
    "reported_opinion_as_own": re.compile(
        r"Susan Elan Jones.{0,100}"
        r"(?:warned against|opposed|rejected|argued against)"
        r".{0,80}(?:referendum|second vote)",
        re.I | re.S,
    ),

    "cross_speaker_phrase": re.compile(
        r"Kwasi Kwarteng.{0,120}"
        r"(?:step backwards|soft way out)",
        re.I | re.S,
    ),

    "chapman_stance_reversal": re.compile(
        r"Jenny Chapman.{0,90}"
        r"(?:opposed|rejected|argued against)"
        r".{0,70}(?:referendum|public vote|ballot)",
        re.I | re.S,
    ),

    "may_stance_reversal": re.compile(
        r"Theresa May.{0,90}"
        r"(?:supported|endorsed|advocated|called for)"
        r".{0,70}(?:second referendum|people['’]s vote)",
        re.I | re.S,
    ),
}


def load_response(path: Path) -> dict:
    """Read JSON, including JSON inside Markdown fences."""

    raw = path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if raw.startswith("```"):
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
        )
        raw = re.sub(
            r"\s*```\s*$",
            "",
            raw,
        )

    result = json.loads(raw)

    if not isinstance(result, dict):
        raise ValueError(
            "Expected a JSON response object."
        )

    return result


def check_response(response: dict) -> dict:
    """Inspect one FastAPI response."""

    issues = []

    question = str(
        response.get("question") or ""
    )

    answer = response.get("answer")

    evidence = response.get("evidence")

    if not question:
        issues.append(
            "Missing question."
        )

    if not isinstance(answer, str) or not answer.strip():
        issues.append(
            "Missing answer."
        )
        answer = ""

    if not isinstance(evidence, list):
        issues.append(
            "Evidence is not a list."
        )
        evidence = []

    # ------------------------------------------------
    # CHECK CITATIONS
    # ------------------------------------------------

    citations = [
        int(value)
        for value in CITATION.findall(answer)
    ]

    invalid = sorted({
        number
        for number in citations
        if number > len(evidence)
    })

    if invalid:
        issues.append(
            f"Invalid citation indices: {invalid}"
        )

    if evidence and not citations:
        issues.append(
            "Answer contains no evidence citations."
        )

    # ------------------------------------------------
    # CHECK EVIDENCE FIELDS
    # ------------------------------------------------

    year_counts = Counter()
    speakers = set()

    for index, item in enumerate(
        evidence,
        1,
    ):

        if not isinstance(item, dict):
            issues.append(
                f"E{index} is not an object."
            )
            continue

        for field in (
            "speaker",
            "date",
            "text",
        ):
            value = item.get(field)

            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                issues.append(
                    f"E{index} missing {field}."
                )

        year = str(
            item.get("date") or ""
        )[:4]

        if re.fullmatch(
            r"20\d{2}",
            year,
        ):
            year_counts[year] += 1

        speaker = str(
            item.get("speaker") or ""
        )

        if speaker:
            speakers.add(speaker)

    # ------------------------------------------------
    # CHECK TEMPORAL COVERAGE
    # ------------------------------------------------

    if SECOND_VOTE.search(question):

        match = YEAR_RANGE.search(
            question
        )

        if match:

            start, end = map(
                int,
                match.groups(),
            )

            if 2015 <= start <= end <= 2019:

                missing = [
                    year
                    for year in range(
                        start,
                        end + 1,
                    )
                    if str(year) not in year_counts
                ]

                if missing:

                    issues.append(
                        "Missing evidence years: "
                        + ", ".join(
                            map(str, missing)
                        )
                    )

    # ------------------------------------------------
    # CHECK PREVIOUSLY OBSERVED ERRORS
    # ------------------------------------------------

    if SECOND_VOTE.search(question):

        for name, pattern in KNOWN_ERRORS.items():

            if pattern.search(answer):

                issues.append(
                    f"Potential attribution error: {name}"
                )

    # ------------------------------------------------
    # RETURN DIAGNOSTIC RESULT
    # ------------------------------------------------

    return {
        "question": question,
        "automated_pass": len(issues) == 0,
        "issues": issues,
        "evidence_count": len(evidence),
        "citation_count": len(citations),
        "unique_citations": sorted(
            set(citations)
        ),
        "year_counts": dict(
            sorted(year_counts.items())
        ),
        "speaker_count": len(speakers),
        "answer": answer,
        "manual_review_required": True,
    }


def call_api(
    base_url: str,
    question: str,
    timeout: int,
) -> dict:
    """Call the running local FastAPI server."""

    url = (
        base_url.rstrip("/")
        + "/graphrag/ask"
    )

    body = json.dumps({
        "question": question
    }).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:

        result = json.load(response)

    if not isinstance(result, dict):
        raise ValueError(
            "API did not return a JSON object."
        )

    return result


def choose_questions(
    path: Path,
    limit: int,
    run_all: bool,
) -> list[str]:
    """Build a varied test set from existing questions."""

    questions_data = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    selected = [
        REFERENDUM_QUESTION
    ]

    # Important existing benchmark questions:
    # simple, comparison, multi-hop, second referendum.
    priority_ids = [
        1,
        3,
        6,
        30,
        37,
    ]

    for question_id in priority_ids:

        for item in questions_data:

            if item.get("id") == question_id:

                question = item.get(
                    "question"
                )

                if (
                    isinstance(question, str)
                    and question not in selected
                ):
                    selected.append(question)

                break

    # Add more questions from each reasoning group.

    groups = {
        "1-hop": [],
        "2-hop": [],
        "3-hop": [],
    }

    for item in questions_data:

        hop = item.get("hop")

        question = item.get(
            "question"
        )

        if (
            hop in groups
            and isinstance(question, str)
            and question not in selected
        ):
            groups[hop].append(
                question
            )

    while any(groups.values()):

        for hop in groups:

            if groups[hop]:

                selected.append(
                    groups[hop].pop(0)
                )

    if run_all:
        return selected

    return selected[:limit]


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Final GraphRAG acceptance checks"
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--saved",
        type=Path,
    )

    mode.add_argument(
        "--live",
        action="store_true",
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/questions.json"),
    )

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/api_acceptance_report.json"
        ),
    )

    args = parser.parse_args()

    reports = []

    # ------------------------------------------------
    # SAVED RESPONSE MODE
    # ------------------------------------------------

    if args.saved:

        response = load_response(
            args.saved
        )

        reports.append(
            check_response(response)
        )

    # ------------------------------------------------
    # LIVE API MODE
    # ------------------------------------------------

    else:

        questions = choose_questions(
            args.questions,
            max(1, args.limit),
            args.all,
        )

        for index, question in enumerate(
            questions,
            1,
        ):

            print(
                f"[{index}/{len(questions)}] "
                f"{question}",
                flush=True,
            )

            try:

                response = call_api(
                    args.url,
                    question,
                    args.timeout,
                )

                reports.append(
                    check_response(response)
                )

            except (
                urllib.error.URLError,
                TimeoutError,
                ValueError,
                OSError,
            ) as error:

                reports.append({
                    "question": question,
                    "automated_pass": False,
                    "issues": [
                        str(error)
                    ],
                    "answer": "",
                })

    # ------------------------------------------------
    # WRITE REPORT
    # ------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            {
                "results": reports
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    passed = 0

    for report in reports:

        if report["automated_pass"]:
            passed += 1
            status = "PASS"
        else:
            status = "REVIEW"

        print(
            f"{status}: {report['question']}"
        )

        for issue in report["issues"]:

            print(
                f"  - {issue}"
            )

    print(
        f"\nAutomated checks: "
        f"{passed}/{len(reports)} passed"
    )

    print(
        f"Report: {args.output}"
    )

    print(
        "Important: structural PASS does not "
        "prove historical factual accuracy."
    )

    return (
        0
        if passed == len(reports)
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
