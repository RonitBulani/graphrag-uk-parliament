# Generate pooled reference answers via TREC-style pooling
# For each question, the reference answer is generated from the UNION of chunks
# retrieved by both Naive RAG and GraphRAG to ensure fairness on reference metrics.

import json
import time
from pathlib import Path

import networkx as nx
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

_ROOT          = Path(__file__).resolve().parent.parent
NAIVE_FILE     = _ROOT / "results" / "naive_rag_results.json"
GRAPH_FILE     = _ROOT / "results" / "graphrag_results.json"
KG_FILE        = _ROOT / "data" / "knowledge_graph.graphml"
QUESTIONS_FILE = _ROOT / "data" / "questions.json"
OUTPUT_FILE    = _ROOT / "results" / "pooled_references.json"

OLLAMA_URL   = "http://localhost:11434/api/generate"
LLM_MODEL    = "qwen2.5:3b"
TOP_SPEAKERS = 3

PARTY_KEYWORDS = {
    "Con":    ["conservative", "tory", "tories", "theresa may", "boris johnson",
               "david davis", "jacob rees-mogg", "erg", "cabinet"],
    "Lab":    ["labour", "jeremy corbyn", "keir starmer", "shadow"],
    "SNP":    ["snp", "scottish national", "scotland", "scottish"],
    "LibDem": ["liberal democrat", "lib dem", "libdem"],
    "DUP":    ["dup", "democratic unionist"],
    "PC":     ["plaid cymru", "wales", "welsh"],
    "Green":  ["green party"],
}

TOPIC_KEYWORDS = {
    "article_50":          ["article 50"],
    "brexit":              ["brexit", "leaving the eu", "leave the eu"],
    "customs_union":       ["customs union"],
    "single_market":       ["single market"],
    "backstop":            ["backstop", "northern ireland protocol", "irish border"],
    "freedom_of_movement": ["freedom of movement", "free movement"],
    "second_referendum":   ["second referendum", "people's vote", "peoples vote"],
    "erg":                 ["european research group", "erg"],
    "dup":                 ["dup", "democratic unionist"],
    "chequers":            ["chequers"],
    "no_deal":             ["no-deal", "no deal", "wto terms", "wto trading"],
    "good_friday":         ["good friday", "belfast agreement"],
    "independence":        ["scottish independence", "independence referendum"],
    "transition":          ["transition period"],
    "immigration":         ["immigration", "migration"],
    "sovereignty":         ["sovereignty", "parliamentary sovereignty"],
    "nhs":                 ["nhs", "national health service", "public services"],
    "trade":               ["trade deal", "trade deals", "trade agreement"],
    "meaningful_vote":     ["meaningful vote", "meaningful votes"],
    "devolved":            ["devolved", "devolution", "holyrood", "cardiff"],
    "cabinet":             ["cabinet resignation", "cabinet resigns", "resigned from cabinet"],
}


# =============================================================================
# ENTITY DETECTION HELPERS
# =============================================================================

# identify which parties should be expected in the answer
def detect_parties(question: str) -> list[str]:
    q = question.lower()
    found = [p for p, kws in PARTY_KEYWORDS.items() if any(kw in q for kw in kws)]
    return found or ["Con", "Lab"]

# identify which topics the question is discussing
def detect_topics(question: str) -> list[str]:
    q = question.lower()
    found = [t for t, kws in TOPIC_KEYWORDS.items() if any(kw in q for kw in kws)]
    return found or ["brexit"]

# find the party associated with a speaker by walking the graph
def _speaker_party(G, speaker_node: str) -> str:
    for nb in G.neighbors(speaker_node):
        if G.nodes[nb].get("node_type") == "PARTY":
            return nb
    try:
        for nb in G.predecessors(speaker_node):
            if G.nodes[nb].get("node_type") == "PARTY":
                return nb
    except Exception:
        pass
    return ""

# find the most relevant speakers for the detected topics using graph edge weights
def kg_top_speakers(G, topics: list[str], parties: list[str], top_n: int) -> list[str]:
    speaker_scores: dict[str, float] = {}
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") != "SPOKE_ABOUT":
            continue
        topic_label = v.lower()
        if not any(any(kw in topic_label for kw in TOPIC_KEYWORDS.get(t, [t])) for t in topics):
            continue
        
        weight = float(data.get("speech_count", 1.0))
        speaker_name  = u
        speaker_party = _speaker_party(G, u)
        
        if parties and not any(p.lower() in speaker_party.lower() for p in parties):
            continue
        speaker_scores[speaker_name] = speaker_scores.get(speaker_name, 0.0) + weight

    # fallback: ignore party filter if no speakers found
    if not speaker_scores:
        for u, v, data in G.edges(data=True):
            if data.get("edge_type") != "SPOKE_ABOUT":
                continue
            topic_label = v.lower()
            if not any(any(kw in topic_label for kw in TOPIC_KEYWORDS.get(t, [t])) for t in topics):
                continue
            weight = float(data.get("speech_count", 1.0))
            speaker_scores[u] = speaker_scores.get(u, 0.0) + weight

    ranked = sorted(speaker_scores.items(), key=lambda x: -x[1])
    return [name for name, _ in ranked[:top_n]]


# =============================================================================
# TREC POOLING & LLM GENERATION
# =============================================================================

# deduplicated union of both systems' retrieved chunk texts
def trec_pool(naive_result: dict, graph_result: dict) -> list[str]:
    seen: set[str] = set()
    pool: list[str] = []
    for chunk in (naive_result.get("retrieved_chunks", []) +
                  graph_result.get("retrieved_chunks", [])):
        text = chunk.get("text", "").strip()
        key  = text[:120]
        if key and key not in seen:
            seen.add(key)
            pool.append(text)
    return pool

# basic generation wrapper for ollama
def llm_generate(prompt: str, max_tokens: int = 350) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
                  "options": {"num_predict": max_tokens, "temperature": 0.1}},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"

# draft the final reference answer using the pooled context
def generate_reference_answer(question: str, pool: list[str]) -> str:
    context = "\n\n".join(pool)
    prompt = (
        "You are a neutral political analyst. Using ONLY the parliamentary speeches "
        "below as your source, write a concise factual answer (3-6 sentences) to the "
        "question. Do not add opinions or information from outside the speeches.\n\n"
        f"Question: {question}\n\nSpeeches:\n{context}\n\nAnswer:"
    )
    return llm_generate(prompt)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("Loading RAG result files for TREC pooling...")
    with open(NAIVE_FILE, encoding="utf-8") as f:
        naive_results = {r["question_id"]: r for r in json.load(f)}
    with open(GRAPH_FILE, encoding="utf-8") as f:
        graph_results = {r["question_id"]: r for r in json.load(f)}
    print(f"  Naive RAG: {len(naive_results)} questions")
    print(f"  GraphRAG:  {len(graph_results)} questions")

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    print("Loading knowledge graph for speaker derivation...")
    G = nx.read_graphml(KG_FILE)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    pooled_references: dict[str, dict] = {}

    for q in questions:
        qid   = q["id"]
        qtext = q["question"]
        hop   = q["hop"]

        if qid not in naive_results or qid not in graph_results:
            print(f"  Q{qid:02d} SKIPPED — missing from result files")
            continue

        print(f"\n  Q{qid:02d} [{hop}]: {qtext[:70]}...")

        parties  = detect_parties(qtext)
        topics   = detect_topics(qtext)
        speakers = kg_top_speakers(G, topics, parties, TOP_SPEAKERS)

        # create the pooled context
        pool = trec_pool(naive_results[qid], graph_results[qid])
        n_chunks = len(naive_results[qid].get("retrieved_chunks", []))
        g_chunks = len(graph_results[qid].get("retrieved_chunks", []))
        print(f"    Pool: {len(pool)} unique chunks (N={n_chunks} + G={g_chunks})")

        # generate the answer
        ref_answer = generate_reference_answer(qtext, pool)
        print(f"    Reference ({len(ref_answer)} chars) | parties={parties}")

        pooled_references[str(qid)] = {
            "question":         qtext,
            "hop":              hop,
            "target_parties":   parties,
            "target_topics":    topics,
            "target_speakers":  speakers,
            "reference_answer": ref_answer,
            "pool_size":        len(pool),
        }

        time.sleep(0.3)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pooled_references, f, indent=2, ensure_ascii=False)
    print(f"\nPooled references written → {OUTPUT_FILE}  ({len(pooled_references)} questions)")


if __name__ == "__main__":
    main()