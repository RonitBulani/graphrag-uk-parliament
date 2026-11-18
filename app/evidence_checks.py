
import re


PARTY_PATTERNS = {
    "SNP": [
        r"\bSNP\b",
        r"\bScottish National Party\b",
    ],
    "DUP": [
        r"\bDUP\b",
        r"\bDemocratic Unionist Party\b",
    ],
    "Con": [
        r"\bConservative(?:s)?\b",
        r"\bTories\b",
        r"\bTory\b",
    ],
    "Lab": [
        r"\bLabour\b",
    ],
}


def check_evidence_coverage(question, chunks):
    """
    Check whether final retrieved evidence covers
    explicitly requested temporal endpoints and
    political parties in comparison questions.

    This is a structural check, not a semantic
    relevance or factual accuracy assessment.
    """

    # Identify explicit year ranges.
    year_match = re.search(
        r"\b(?:between|from)\s+(20\d{2})\s+"
        r"(?:and|to)\s+(20\d{2})\b",
        question,
        flags=re.IGNORECASE,
    )

    requested_years = []

    if year_match:
        requested_years = [
            year_match.group(1),
            year_match.group(2),
        ]

    # Extract actual years from speech dates.
    evidence_years = {
        str(chunk.get("date", ""))[:4]
        for chunk in chunks
    }

    missing_years = [
        year
        for year in requested_years
        if year not in evidence_years
    ]

    # Identify explicitly named political parties.
    requested_parties = []

    for party, patterns in PARTY_PATTERNS.items():
        if any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in patterns
        ):
            requested_parties.append(party)

    # For comparisons, require evidence from each
    # explicitly named party.
    evidence_parties = {
        chunk.get("party")
        for chunk in chunks
    }

    missing_parties = []

    if len(requested_parties) >= 2:
        missing_parties = [
            party
            for party in requested_parties
            if party not in evidence_parties
        ]

    coverage_ok = (
        bool(chunks)
        and not missing_years
        and not missing_parties
    )

    return {
        "coverage_ok": coverage_ok,
        "requested_years": requested_years,
        "evidence_years": sorted(evidence_years),
        "missing_years": missing_years,
        "requested_parties": requested_parties,
        "missing_parties": missing_parties,
    }
