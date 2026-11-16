
import json
from typing import Any


VALID_STANCES = {
    "SUPPORTS",
    "OPPOSES",
    "NOT_CALLING_FOR",
    "UNCLEAR",
}


def validate_evidence_note(
    note: Any,
    chunk: dict,
) -> dict:
    """
    Validate the structure of one extracted note.

    This verifies data types and exact quotations.
    It does not independently prove the interpretation.
    """
    if not isinstance(note, dict):
        raise ValueError("Evidence note must be a JSON object.")

    required = {
        "speaker_position",
        "positions_attributed_to_others",
        "stance",
        "supporting_quote",
    }

    missing = required - set(note)

    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")

    position = note["speaker_position"]
    others = note["positions_attributed_to_others"]
    stance = note["stance"]
    quote = note["supporting_quote"]

    if not isinstance(position, str) or not position.strip():
        raise ValueError("Invalid speaker_position.")

    if not isinstance(others, list):
        raise ValueError(
            "positions_attributed_to_others must be a list."
        )

    if not all(
        isinstance(item, str) and item.strip()
        for item in others
    ):
        raise ValueError("Invalid attributed position.")

    if stance not in VALID_STANCES:
        raise ValueError(f"Invalid stance: {stance}")

    if (
        not isinstance(quote, str)
        or not quote.strip()
        or quote not in chunk["text"]
    ):
        raise ValueError(
            "Supporting quotation was not found in the source."
        )

    # Ground metadata in the retrieved speech,
    # not in language-model-generated text.
    return {
        "speaker": chunk["speaker"],
        "party": chunk["party"],
        "date": chunk["date"],
        "speaker_position": position.strip(),
        "positions_attributed_to_others": others,
        "stance": stance,
        "supporting_quote": quote,
        "quote_verified": True,
    }


def extract_evidence_note(
    engine,
    chunk: dict,
) -> dict:
    """
    Extract a structured interpretation of one speech.

    Each speech is analysed independently.
    """

    prompt = f"""
You are analysing ONE parliamentary speech excerpt.

Identify the speaker's own argument and distinguish it
from opinions or proposals attributed to other people.

Return ONLY a valid JSON object with this structure:

{{
  "speaker_position": "One sentence describing the speaker's own argument",
  "positions_attributed_to_others": [
    "A position the speaker attributes to someone else"
  ],
  "stance": "SUPPORTS",
  "supporting_quote": "Exact phrase from the excerpt"
}}

STANCE must be one of:
SUPPORTS
OPPOSES
NOT_CALLING_FOR
UNCLEAR

Rules:
1. speaker_position must describe the speaker's own view,
   not merely repeat what other people think.
2. When a speaker criticises a proposal attributed to
   others, do not describe the speaker as endorsing it.
3. Use OPPOSES for clear opposition or criticism.
4. Use NOT_CALLING_FOR when the speaker says they are
   not advocating another vote without clearly opposing it.
5. Use UNCLEAR if the speaker's own position is uncertain.
6. Put views attributed to other people in
   positions_attributed_to_others.
7. If there are no such attributed views, return [].
8. supporting_quote must be copied exactly from the
   source and support the speaker's own interpretation.
9. Do not invent context beyond the excerpt.
10. Do not add extra commentary outside the JSON object.

Speaker: {chunk["speaker"]}
Party: {chunk["party"]}
Date: {chunk["date"]}

Speech excerpt:
{chunk["text"]}
"""

    response = engine.llm_client.chat.completions.create(
        model=engine.ollama_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        max_tokens=500,
    )

    raw = response.choices[0].message.content

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Empty evidence-extraction response.")

    raw = raw.strip()

    # Accept occasional Markdown code fences.
    if raw.startswith("```"):
        lines = raw.splitlines()

        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()

    try:
        note = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Evidence extraction returned invalid JSON: {raw}"
        ) from exc

    return validate_evidence_note(note, chunk)
