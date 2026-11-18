"""Lightweight grounded generation, v4.

Drop-in replacement for app/general_grounded_generation.py.
The FastAPI service still calls generate_general_grounded_answer(engine, question, chunks, plan).

Modes:
  GRAPHRAG_GENERATION_MODE=extractive  (default; zero extra LLM/NLI calls)
  GRAPHRAG_GENERATION_MODE=verified    (Qwen drafts; only NLI-accepted claims used)
  GRAPHRAG_NLI_ENABLED=1              (required for accepting generated claims)
  GRAPHRAG_SYNTHESIS_MODEL=qwen2.5:3b (reuse engine's Ollama model)
  GRAPHRAG_GENERATION_TRACE=data/general_generation_trace.jsonl

No change to the research engine, vector store, or temporal generator.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.question_aware_retrieval import plan_question

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
TRACE_LOCK = threading.Lock()

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u201c])")
WORDS = re.compile(r"[a-z][a-z0-9'-]{2,}", re.I)
STOP = set("the and with from that this would could should there their these those have what which about into only also then when been will want very party member house europe european union labour conservative brexit leaving leave".split())

REPORTED = re.compile(
    r"\b(?:the leave campaign|the argument made|opponents? (?:say|argue|might)|"
    r"we (?:are|were) told|some (?:members|people) (?:say|argue)|"
    r"i thought the dup'?s objection|the dup'?s objection|"
    r"those who argued|does my hon\.? friend not find)\b", re.I
)
CHALLENGE = re.compile(
    r"\b(?:i find it remarkable|find it curious|a nonsense|"
    r"misleading|however|yet|and yet|do not support|does not support)\b", re.I
)
REMAIN = re.compile(
    r"\b(?:we should remain|should stay in the european union|"
    r"best interests to stay within the european union|i will vote remain)\b", re.I
)
LEAVE = re.compile(
    r"\b(?:we should leave|i support leaving|if we left|"
    r"more prosperous if we left|we will take back control|"
    r"i want that to happen as we leave|independent trade policy|"
    r"we (?:will|would) be able to shape trade policy|"
    r"regaining control of our own borders|"
    r"one of the key reasons for doing so|end free movement)\b", re.I
)
LAB_POLICY = re.compile(
    r"\b(?:labour (?:would|will|supports|accepts|wants)|"
    r"labour'?s (?:policy|position|approach)|"
    r"permanent (?:and comprehensive )?customs union|"
    r"our manifesto|our policy|we will prioritise jobs)\b", re.I
)
CON_POLICY = re.compile(
    r"\b(?:we are leaving (?:the )?(?:single market|customs union)|"
    r"we are going to have an independent trade policy|"
    r"the prime minister has been clear that we are leaving|"
    r"the government(?:'s)? (?:policy|objectives)|"
    r"our independent trade policy)\b", re.I
)
PURPOSE = re.compile(
    r"\b(?:need for a backstop|backstop to (?:ensure|protect|prevent)|"
    r"backstop was there to guarantee|legally binding backstop|"
    r"guarantee that there be no hard border|no hard border)\b", re.I
)
OBJECTION = re.compile(
    r"\b(?:backstop risks|rips up the union|"
    r"trapped in an arrangement|no unilateral exit|"
    r"barrier down the irish sea|"
    r"checks on trade between northern ireland and great britain|"
    r"regulatory aligned|regulation without representation|"
    r"undemocratic|a threat to our precious union|"
    r"regulatory alignment)\b", re.I
)
PERSONAL = re.compile(
    r"\b(?:i (?:do not believe|believe|support|oppose|want)|"
    r"my heart lies|i would support|i cannot support|"
    r"we should not|second referendum would not|"
    r"labour will support or table|labour would support)\b", re.I
)


def _terms(text: str) -> set[str]:
    return {x.lower() for x in WORDS.findall(text) if x.lower() not in STOP}


def _sentences(text: str) -> list[str]:
    # Do not silently 'repair' damaged source text: quoted text remains traceable.
    raw = [s.strip() for s in SENTENCE.split(text) if s.strip()]
    result = []
    for sentence in raw:
        if result and re.search(r"\b(?:hon|Mr|Mrs|Ms|Dr|Prof|St)\.$", result[-1], re.I):
            result[-1] += " " + sentence
        else:
            result.append(sentence)
    return result


def _wrong_topic(question: str, chunk: dict) -> bool:
    q = question.lower()
    if "referendum" not in q or "scottish" in q or "scotland" in q:
        return False
    agenda = str(chunk.get("agenda", "")).lower()
    text = str(chunk.get("text", "")).lower()
    return "claim of right for scotland" in agenda or (
        "indyref2" in text and "second eu referendum" not in text
    )


def _facet(chunk: dict, kind: str) -> str:
    current = str(chunk.get("retrieval_facet") or "general")
    if kind != "two_sided_explanation":
        return current
    text = str(chunk.get("text", ""))
    purpose, objection = bool(PURPOSE.search(text)), bool(OBJECTION.search(text))
    if purpose and not objection:
        return "purpose"
    if objection and not purpose:
        return "objections"
    return current


def _rank(chunk: dict, kind: str, facet: str) -> float:
    text = str(chunk.get("text", ""))
    party = str(chunk.get("party", ""))
    score = 0.10 * float(chunk.get("rerank_score") or 0)
    if kind == "arguments_for_leaving":
        score += 9 if LEAVE.search(text) else -8
        score -= 12 if REMAIN.search(text) else 0
        score -= 9 if REPORTED.search(text) and CHALLENGE.search(text) else 0
        # Rhetorical 'if we left, would ...?' is not advocacy.
        score -= 12 if re.search(r"if we left.{0,110}\?", text, re.I | re.S) else 0
    elif kind == "party_comparison":
        own = LAB_POLICY if party == "Lab" else CON_POLICY
        score += 10 if own.search(text) else -4
        if party == "Lab" and re.search(r"labour.{0,100}(?:customs union|customs arrangement)", text, re.I | re.S):
            score += 4
        if party == "Con" and re.search(r"\bwe are leaving the (?:customs union|single market)\b", text, re.I):
            score += 6
        if party == "Con" and "labour" in text.lower():
            score -= 7
        if "€" in text:
            score -= 8
    elif kind == "role_comparison":
        own = LAB_POLICY if facet == "labour_policy" else PERSONAL
        score += 10 if own.search(text) else -5
        if facet == "labour_policy":
            score += 7 if re.search(r"\blabour (?:will|would) support\b", text, re.I) else 0
            score -= 8 if text.rstrip().endswith("?") else 0
        score += 3 if re.search(r"second referendum|confirmatory|public vote", text, re.I) else -5
    elif kind == "two_sided_explanation":
        if facet == "purpose":
            score += 10 if PURPOSE.search(text) else -12
            score -= 8 if OBJECTION.search(text) and not PURPOSE.search(text) else 0
        else:
            score += 10 if OBJECTION.search(text) else -12
            score += 4 if party == "DUP" and OBJECTION.search(text) else 0
            score -= 8 if REPORTED.search(text) else 0
    return score


def _extract_sentence(chunk: dict, question: str, kind: str, facet: str) -> str:
    """Choose source-owned text; never synthesize quotes."""
    text = str(chunk.get("text", ""))
    sentences = _sentences(text)
    if not sentences:
        return ""
    qwords = _terms(question)
    scored = []
    for i, sentence in enumerate(sentences):
        if len(sentence) < 35:
            continue
        score = 1.5 * len(_terms(sentence) & qwords)
        if kind == "arguments_for_leaving":
            score += 11 if LEAVE.search(sentence) else -4
            score -= 14 if REMAIN.search(sentence) else 0
            score -= 7 if REPORTED.search(sentence) else 0
            score -= 12 if re.search(r"if we left.{0,110}\?", sentence, re.I | re.S) else 0
        elif kind == "party_comparison":
            own = LAB_POLICY if chunk.get("party") == "Lab" else CON_POLICY
            score += 10 if own.search(sentence) else -6
            if chunk.get("party") == "Lab" and re.search(r"labour.{0,100}customs union", sentence, re.I | re.S):
                score += 5
            if chunk.get("party") == "Con" and re.search(r"we are leaving the (?:customs union|single market)", sentence, re.I):
                score += 5
            score -= 7 if "€" in sentence else 0
            score += 3 if "customs union" in sentence.lower() else 0
            score -= 7 if chunk.get("party") == "Con" and "labour" in sentence.lower() and not CON_POLICY.search(sentence) else 0
        elif kind == "role_comparison":
            pattern = LAB_POLICY if facet == "labour_policy" else PERSONAL
            score += 10 if pattern.search(sentence) else -5
            if facet == "labour_policy" and re.search(r"labour (?:will|would) support", sentence, re.I):
                score += 8
            score -= 7 if sentence.rstrip().endswith("?") else 0
            score += 4 if re.search(r"second referendum|public vote|confirmatory", sentence, re.I) else 0
        elif kind == "two_sided_explanation":
            want = PURPOSE if facet == "purpose" else OBJECTION
            other = OBJECTION if facet == "purpose" else PURPOSE
            score += 12 if want.search(sentence) else -9
            score -= 8 if other.search(sentence) and not want.search(sentence) else 0
            if facet == "objections" and REPORTED.search(sentence):
                score -= 5
        scored.append((score, -i, i))
    if not scored:
        return ""
    best_score, _, best_i = max(scored)
    if best_score < -3:
        return ""
    excerpt = sentences[best_i]
    # Resolves anaphora like 'That is where my heart lies'.
    if best_i > 0 and re.match(r"^(?:that is where|i want that|this means|that would)\b", excerpt, re.I):
        excerpt = sentences[best_i - 1] + " " + excerpt
    if best_i + 1 < len(sentences) and len(excerpt) < 105:
        next_sentence = sentences[best_i + 1]
        if len(excerpt) + len(next_sentence) + 1 <= 470:
            excerpt += " " + next_sentence
    return excerpt if len(excerpt) <= 550 else sentences[best_i]


def _safe_fallback(chunk: dict, question: str, kind: str, facet: str) -> str:
    text = str(chunk.get("text", ""))
    if kind == "arguments_for_leaving":
        if not LEAVE.search(text) or REMAIN.search(text) or (REPORTED.search(text) and CHALLENGE.search(text)):
            return ""
        if re.search(r"if we left.{0,110}\?", text, re.I | re.S):
            return ""
    excerpt = _extract_sentence(chunk, question, kind, facet)
    if not excerpt:
        return ""
    if kind == "arguments_for_leaving" and (not LEAVE.search(excerpt) or REMAIN.search(excerpt) or (REPORTED.search(excerpt) and CHALLENGE.search(excerpt))):
        return ""
    if kind == "two_sided_explanation":
        want = PURPOSE if facet == "purpose" else OBJECTION
        if not want.search(excerpt):
            return ""
        if facet == "objections" and REPORTED.search(excerpt):
            return ""  # Prefer the actual objector; description != endorsement.
    speaker = str(chunk.get("speaker", "Unknown speaker"))
    date = str(chunk.get("date", ""))
    return f"{speaker} ({date}) said: “{excerpt}”"


def _windows(rows: list[tuple[int, dict, str]], question: str, kind: str) -> dict[int, str]:
    windows = {}
    for i, chunk, facet in rows:
        excerpt = _extract_sentence(chunk, question, kind, facet)
        if excerpt:
            windows[i] = excerpt[:700]
    return windows


def _qwen_drafts(engine, question: str, rows: list[tuple[int, dict, str]], windows: dict[int, str]) -> list[dict]:
    model = os.getenv("GRAPHRAG_SYNTHESIS_MODEL", "").strip() or getattr(engine, "ollama_model", "qwen2.5:3b")
    material = "\n\n".join(
        f"[E{i+1}] {c.get('speaker')} ({c.get('party')}, {c.get('date')}); aspect={facet}\n{windows[i]}"
        for i, c, facet in rows if i in windows
    )
    prompt = (
        "Use ONLY the parliamentary excerpts below. Output one JSON object with a 'claims' array. "
        "Each claim needs exactly two fields: evidence_id (e.g. E2) and claim (one factual sentence naming the cited speaker). "
        "Do not invent facts. Separate an MP's own view from arguments they quote, describe or criticise. "
        "Do not confuse a Scottish independence referendum with a Brexit referendum. "
        "Do not claim anyone is a frontbencher or backbencher without explicit dated evidence. "
        "Up to 6 claims. Return an empty array if unsupported.\n\n"
        f"QUESTION:\n{question}\n\nEVIDENCE:\n{material}"
    )
    response = engine.llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=650,
        # Native Ollama models may remain resident; use same small model as R2.
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("claims"), list):
        raise ValueError("Expected JSON object with claims list")
    return parsed["claims"]


def _validate_drafts(drafts: list, chunks: list[dict], eligible_ids: set[int], windows: dict[int, str], trace: dict) -> list[dict]:
    """Python owns all span IDs and facet IDs. The model must not supply them."""
    accepted, seen = [], set()
    for row in drafts:
        reason = ""
        if not isinstance(row, dict):
            reason = "invalid_record"
        else:
            match = re.fullmatch(r"E([1-9]\d*)", str(row.get("evidence_id", "")))
            index = int(match.group(1)) - 1 if match else -1
            claim = str(row.get("claim") or "").strip()
            if index not in eligible_ids or index not in windows:
                reason = "invalid_evidence_id"
            elif index in seen:
                reason = "duplicate_evidence_id"
            elif not 15 <= len(claim) <= 320:
                reason = "invalid_claim_length"
            elif str(chunks[index].get("speaker", "")).lower() not in claim.lower():
                reason = "speaker_mismatch"
            elif re.search(r"\b\d{4}\b", claim) and any(
                y not in str(chunks[index].get("text", "")) and y not in str(chunks[index].get("date", ""))
                for y in re.findall(r"\b\d{4}\b", claim)
            ):
                reason = "new_year_in_claim"
            else:
                accepted.append({"evidence_id": f"E{index+1}", "span_id": "S1", "claim": claim})
                seen.add(index)
        if reason:
            trace["rejected"][reason] += 1
    trace["claims_valid_before_nli"] = len(accepted)
    return accepted


def _nli_approve(claims: list[dict], chunks: list[dict], windows: dict[int, str], trace: dict) -> dict[int, str]:
    if not claims:
        return {}
    if os.getenv("GRAPHRAG_NLI_ENABLED", "0").strip().lower() not in {"1", "true", "yes"}:
        trace["nli_status"] = "disabled; drafts NOT used as verified facts"
        return {}
    try:
        # Heavy verifier is imported / initialized only when explicitly enabled.
        from app.grounded_generation import _verifier, _verify_claims
        verifier, labels = _verifier()
        spans = {f"E{i+1}": {"S1": window} for i, window in windows.items()}
        results = _verify_claims(claims, chunks, spans, verifier, labels)
        allowed = {(int(c["evidence_id"][1:]) - 1, c["claim"]) for c in claims}
        verified = {}
        for row in results:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                i, claim = row[0], str(row[1])
                if isinstance(i, int) and (i, claim) in allowed:
                    verified[i] = claim
        trace["claims_accepted_by_nli"] = len(verified)
        trace["claims_rejected_by_nli"] = len(claims) - len(verified)
        trace["nli_status"] = "completed"
        return verified
    except Exception as ex:
        trace["nli_status"] = f"error: {type(ex).__name__}: {str(ex)[:220]}"
        logger.warning("NLI unavailable; no generated claims promoted: %s", ex)
        return {}


def _intro(kind: str, chunks: list[dict], facets: dict[int, str]) -> str:
    # Avoid claim synthesis here unless directly grounded in identified sources.
    if kind == "role_comparison":
        return ("**Role comparison is provisional.** These speech records identify MPs and parties, "
                "but they do not verify frontbench/backbench appointments on each date.")
    if kind == "party_comparison":
        lab = next((i for i,c in enumerate(chunks) if facets.get(i)=="labour" and "customs union" in str(c.get("text", "")).lower() and LAB_POLICY.search(str(c.get("text", "")))), None)
        con_candidates = [i for i,c in enumerate(chunks) if facets.get(i)=="conservative" and CON_POLICY.search(str(c.get("text", ""))) and re.search(r"\bleav(?:e|ing) the (?:single market|customs union)\b", str(c.get("text", "")), re.I)]
        con = max(con_candidates, key=lambda i: _rank(chunks[i], kind, "conservative"), default=None)
        if lab is not None and con is not None:
            return ("**Policy contrast in the selected speeches:** Labour described pursuing a close trading relationship "
                    f"and customs union [E{lab+1}], while the Conservative government described leaving the "
                    f"customs union [E{con+1}]. These dated statements do not establish party unanimity.")
    if kind == "two_sided_explanation":
        purpose = next((i for i,c in enumerate(chunks) if facets.get(i)=="purpose" and re.search(r"backstop",str(c.get("text","")),re.I) and re.search(r"no hard border",str(c.get("text","")),re.I)), None)
        objections = [i for i,c in enumerate(chunks) if facets.get(i)=="objections" and re.search(r"(?:rips up the union|checks on trade|barrier down the irish sea|split the united kingdom)",str(c.get("text","")),re.I)]
        objection = max(objections, key=lambda i: _rank(chunks[i], kind, "objections"), default=None)
        if purpose is not None and objection is not None:
            return ("The backstop was intended to avoid a hard Irish border if longer-term arrangements were not ready "
                    f"[E{purpose+1}]. Critics feared regulatory separation within the UK [E{objection+1}].")
    return ""


def _assemble(question: str, chunks: list[dict], plan: dict, facets: dict[int, str], verified: dict[int, str], trace: dict) -> str:
    kind = str(plan.get("kind") or "general")
    sections = plan.get("facets") or [{"name":"general", "title":"Relevant parliamentary evidence", "query":question}]
    paragraphs = []
    intro = _intro(kind, chunks, facets)
    if intro:
        paragraphs.append(intro)
    for section in sections:
        name = section["name"]
        indices = [i for i in range(len(chunks)) if facets.get(i) == name]
        indices.sort(key=lambda i: _rank(chunks[i],kind,name) + (3 if i in verified else 0), reverse=True)
        sentences, used_speakers = [], set()
        for i in indices:
            speaker = str(chunks[i].get("speaker", "")).lower()
            if speaker in used_speakers:
                continue
            if i in verified:
                statement = verified[i]
                trace["claims_used"] += 1
            else:
                statement = _safe_fallback(chunks[i], str(section.get("query") or question), kind, name)
                if statement:
                    trace["fallback_quotes_used"] += 1
            if statement:
                sentences.append(f"{statement.rstrip('. ')} [E{i+1}].")
                used_speakers.add(speaker)
            if len(sentences) >= 2:
                break
        paragraphs.append(f"**{section.get('title', name)}**\n\n" + (
            " ".join(sentences) if sentences else "The retrieved passages do not establish this aspect reliably."
        ))
    if kind == "role_comparison":
        paragraphs.append("An authoritative frontbench/backbench comparison requires independently verified dated roles.")
    return "\n\n".join(paragraphs)


def _trace_write(trace: dict) -> None:
    path_str = os.getenv("GRAPHRAG_GENERATION_TRACE", "data/general_generation_trace.jsonl").strip()
    if not path_str or path_str.lower() in {"off", "0", "false"}:
        return
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = dict(trace)
        serializable["rejected"] = dict(serializable["rejected"])
        with TRACE_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
    except OSError as ex:
        logger.warning("Could not write generation trace: %s", ex)


def generate_general_grounded_answer(engine, question: str, final_chunks: list[dict], plan: dict | None = None) -> str:
    """Answer with source-owned excerpts or optionally NLI-verified Qwen claims.

    Warning: Citation IDs refer to final_chunks *without reordering*.
    """
    chunks = list(final_chunks or [])
    plan = plan if plan is not None else plan_question(question)
    kind = str(plan.get("kind", "general"))
    mode = os.getenv("GRAPHRAG_GENERATION_MODE", "extractive").strip().lower()
    if mode not in {"extractive", "verified"}:
        raise ValueError("GRAPHRAG_GENERATION_MODE must be extractive or verified")
    trace = {
        "version": 4, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": question, "kind": kind, "mode": mode,
        "evidence_count": len(chunks), "claims_generated": 0,
        "claims_valid_before_nli": 0, "claims_accepted_by_nli": 0,
        "claims_rejected_by_nli": 0, "claims_used": 0,
        "fallback_quotes_used": 0, "skipped_wrong_topic": 0,
        "rejected": Counter(), "nli_status": "not requested",
    }
    if not chunks:
        _trace_write(trace)
        return "No relevant parliamentary speeches were retrieved."
    facets = {i: _facet(chunk, kind) for i, chunk in enumerate(chunks)}
    eligible = [(i, c, facets[i]) for i,c in enumerate(chunks) if not _wrong_topic(question, c)]
    excluded = set(range(len(chunks))) - {i for i,_,_ in eligible}
    for i in excluded:
        facets[i] = "excluded"
    trace["skipped_wrong_topic"] = len(excluded)
    verified = {}
    if mode == "verified" and eligible:
        try:
            # Small and balanced Qwen context; avoids loading 10 entire speeches.
            windows_all = _windows(eligible, question, kind)
            chosen = []
            grouped = defaultdict(list)
            for i,c,facet in eligible:
                if i in windows_all:
                    grouped[facet].append((i,c,facet))
            for name, group in grouped.items():
                group.sort(key=lambda row: _rank(row[1], kind, name), reverse=True)
                chosen.extend(group[:2])
            chosen.sort(key=lambda row: row[0])
            chosen = chosen[:6]
            windows = {i: windows_all[i] for i,_,_ in chosen}
            drafts = _qwen_drafts(engine, question, chosen, windows)
            trace["claims_generated"] = len(drafts)
            validated = _validate_drafts(drafts, chunks, set(windows), windows, trace)
            verified = _nli_approve(validated, chunks, windows, trace)
        except Exception as ex:
            trace["generation_error"] = f"{type(ex).__name__}: {str(ex)[:220]}"
            logger.warning("Qwen drafting failed, using source excerpts: %s", ex)
    answer = _assemble(question, chunks, plan, facets, verified, trace)
    trace["facet_counts"] = dict(Counter(facets.values()))
    _trace_write(trace)
    return answer
