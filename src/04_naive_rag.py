# Naive RAG baseline: penta-stream retrieval with RRF, cross-encoder rerank, and party diversity
# HyDE docs are pre-generated before BM25 loads so they don't spike RAM together

import json
import csv
import time
import re
import gc
import requests
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from openai import OpenAI
import nltk
from nltk.corpus import stopwords as _nltk_stop

try:
    _STOP_WORDS = set(_nltk_stop.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    _STOP_WORDS = set(_nltk_stop.words('english'))


# =============================================================================
# CONFIGURATION
# =============================================================================

_ROOT            = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR  = _ROOT / "data" / "vectorstore"
QUESTIONS_FILE   = _ROOT / "data" / "questions.json"
RESULTS_FILE     = _ROOT / "results" / "naive_rag_results.json"
RESULTS_CSV      = _ROOT / "results" / "naive_rag_results.csv"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-12-v2"

LLM_PROVIDERS = {
    "ollama_qwen3b":  {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "qwen2.5:3b"},
    "ollama_phi3":    {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "phi3:mini"},
    "ollama_llama3":  {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "llama3.2:3b"},
}

PROVIDER = "ollama_qwen3b"

_prov        = LLM_PROVIDERS[PROVIDER]
LLM_BASE_URL = _prov["base_url"]
LLM_API_KEY  = _prov["api_key"]
OLLAMA_MODEL = _prov["model"]

TOP_K       = 5
RERANK_POOL = 60


# loading questions from shared JSON so all scripts use identical question sets
with open(QUESTIONS_FILE, encoding="utf-8") as _qf:
    _q_data = json.load(_qf)
TEST_QUESTIONS = [q["question"] for q in _q_data]
QUESTION_IDS   = [q["id"]       for q in _q_data]


# =============================================================================
# LOADING MODELS AND VECTOR STORE
# =============================================================================

print("Loading models and vector store...")
print("-" * 60)

model    = SentenceTransformer(EMBEDDING_MODEL)
print(f"  Bi-encoder loaded: {EMBEDDING_MODEL}")

reranker = CrossEncoder(RERANKER_MODEL)
print(f"  Cross-encoder loaded: {RERANKER_MODEL}")

client     = chromadb.PersistentClient(path=VECTORSTORE_DIR)
collection = client.get_collection("brexit_speeches")
n_chunks   = collection.count()
print(f"  Vector store: {n_chunks:,} chunks")


# =============================================================================
# CHECKING LLM CONNECTION
# =============================================================================

print(f"\nChecking LLM connection [{PROVIDER}]...")
print("-" * 60)

llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "none")

try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    models   = [m["name"] for m in response.json().get("models", [])]
    print(f"  Ollama running. Available models: {models}")
    if not any(OLLAMA_MODEL in m for m in models):
        print(f"  WARNING: {OLLAMA_MODEL} not found! Run: ollama pull {OLLAMA_MODEL}")
except requests.ConnectionError:
    print("  ERROR: Cannot connect to Ollama. Run: ollama serve")
    exit(1)


# =============================================================================
# PARTY DETECTION HELPERS
# =============================================================================

# explicit mentions + topic-implied parties
_PARTY_KEYWORDS = {
    "conservative": "Con", "tory": "Con", "tories": "Con", "backbencher": "Con",
    "labour": "Lab",  "corbyn": "Lab",  "starmer": "Lab",
    "snp": "SNP",     "scottish national": "SNP",  "sturgeon": "SNP",
    "lib dem": "LibDem", "liberal democrat": "LibDem",
    "dup": "DUP",     "green": "Green",
}

# without this, "chequers" gets no party hint → HyDE drifts Labour → Con chunks miss top-k
_IMPLICIT_PARTY_KEYWORDS = {
    "chequers":                ["Con"],
    "lancaster house":         ["Con"],
    "european research group": ["Con"],
    " erg ":                   ["Con"],
    "rees-mogg":               ["Con"],
    "boris johnson":           ["Con"],
    "dominic raab":            ["Con"],
    "david davis":             ["Con"],
    "freedom of movement":     ["Lab", "Con", "SNP"],
    "customs union":           ["Lab", "Con"],
    "single market":           ["Lab", "Con"],
    "withdrawal agreement":    ["Con"],
    "no-deal":                 ["Con"],
    "no deal":                 ["Con"],
    "backstop":                ["DUP", "Con"],
    "northern ireland":        ["DUP"],
}


# return party labels detected in the question text
def detect_parties(question):
    q     = question.lower()
    found = set()
    for keyword, label in _PARTY_KEYWORDS.items():
        if keyword in q:
            found.add(label)
    for keyword, labels in _IMPLICIT_PARTY_KEYWORDS.items():
        if keyword in q:
            found.update(labels)
    return list(found)


# =============================================================================
# HYDE GENERATION
# =============================================================================

# generate a hypothetical speech excerpt to close the query-document embedding gap
def generate_hyde(question):
    q = question.lower()
    if any(w in q for w in ["conservative", "tory", "tories", "backbencher",
                             "chequers", "rees-mogg", "boris johnson"]):
        party_hint = " Write this from the perspective of a Conservative MP."
    elif any(w in q for w in ["labour", "corbyn", "starmer"]):
        party_hint = " Write this from the perspective of a Labour MP."
    elif any(w in q for w in ["snp", "scottish national", "sturgeon"]):
        party_hint = " Write this from the perspective of an SNP MP."
    elif any(w in q for w in ["lib dem", "liberal democrat"]):
        party_hint = " Write this from the perspective of a Liberal Democrat MP."
    elif any(w in q for w in ["dup", "northern ireland", "backstop"]):
        party_hint = " Write this from the perspective of a DUP or Conservative MP."
    else:
        party_hint = ""

    for attempt in range(3):
        try:
            resp = llm_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": (
                    f"Write a short excerpt from a UK parliamentary speech (2-3 sentences) "
                    f"that directly addresses this question:\n\n{question}\n\n"
                    f"{party_hint}Speech excerpt:"
                )}],
                temperature=0.0,
                max_tokens=150,
                timeout=45,
            )
            return resp.choices[0].message.content or question
        except Exception as e:
            if attempt < 2:
                print(f" [HyDE retry {attempt+1}: {type(e).__name__}]", end=" ", flush=True)
                time.sleep(5)
            else:
                print(f" [HyDE failed, using question as fallback]", end=" ", flush=True)
                return question


# =============================================================================
# RETRIEVAL FUNCTION
# =============================================================================

# run penta-stream retrieval and return top_k reranked chunks
def retrieve(question, top_k=TOP_K, hyde_doc=None):
    hypothetical_doc = hyde_doc if hyde_doc is not None else generate_hyde(question)
    hyde_embedding   = model.encode([hypothetical_doc]).tolist()

    # stream 1: HyDE dense
    dense_results    = collection.query(query_embeddings=hyde_embedding,
                                        n_results=min(RERANK_POOL, n_chunks))
    dense_candidates = {}
    for rank in range(len(dense_results["ids"][0])):
        doc_id = dense_results["ids"][0][rank]
        dense_candidates[doc_id] = {
            "text":      dense_results["documents"][0][rank],
            "speaker":   dense_results["metadatas"][0][rank]["speaker"],
            "party":     dense_results["metadatas"][0][rank]["party"],
            "date":      dense_results["metadatas"][0][rank]["date"],
            "agenda":    dense_results["metadatas"][0][rank]["agenda"],
            "distance":  dense_results["distances"][0][rank],
            "rrf_dense": 1.0 / (60 + rank),
        }

    # stream 2: direct question embedding (safety net when HyDE drifts)
    direct_embedding  = model.encode([question]).tolist()
    direct_results    = collection.query(query_embeddings=direct_embedding,
                                         n_results=min(RERANK_POOL, n_chunks))
    direct_candidates = {}
    for rank in range(len(direct_results["ids"][0])):
        doc_id = direct_results["ids"][0][rank]
        direct_candidates[doc_id] = {
            "text":       direct_results["documents"][0][rank],
            "speaker":    direct_results["metadatas"][0][rank]["speaker"],
            "party":      direct_results["metadatas"][0][rank]["party"],
            "date":       direct_results["metadatas"][0][rank]["date"],
            "agenda":     direct_results["metadatas"][0][rank]["agenda"],
            "distance":   direct_results["distances"][0][rank],
            "rrf_direct": 1.0 / (60 + rank),
        }

    # stream 3: BM25 -- catches exact keyword matches that embeddings sometimes miss
    bm25_query_tokens = [w for w in question.lower().split() if w not in _STOP_WORDS]
    bm25_scores       = bm25.get_scores(bm25_query_tokens)
    max_bm25          = float(bm25_scores.max()) if bm25_scores.max() > 0 else 1.0
    bm25_top_idx      = np.argsort(bm25_scores)[::-1][:RERANK_POOL]
    bm25_top_filtered = [idx for idx in bm25_top_idx if bm25_scores[idx] > 0]

    bm25_top_ids = [ALL_IDS[idx] for idx in bm25_top_filtered]
    bm25_doc_map = {}
    if bm25_top_ids:
        fetched      = collection.get(ids=bm25_top_ids, include=["documents"])
        bm25_doc_map = dict(zip(fetched["ids"], fetched["documents"]))

    bm25_candidates = {}
    for rank, idx in enumerate(bm25_top_filtered):
        doc_id = ALL_IDS[idx]
        bm25_candidates[doc_id] = {
            "text":     bm25_doc_map.get(doc_id, ""),
            "speaker":  ALL_METAS[idx]["speaker"],
            "party":    ALL_METAS[idx]["party"],
            "date":     ALL_METAS[idx]["date"],
            "agenda":   ALL_METAS[idx]["agenda"],
            "distance": 1.0 - bm25_scores[idx] / max_bm25,
            "rrf_bm25": 1.0 / (60 + rank),
        }

    # stream 4: party-targeted HyDE -- guarantees named-party chunks regardless of HyDE direction
    party_candidates = {}
    target_parties   = detect_parties(question)
    per_party_n      = max(20, RERANK_POOL // max(len(target_parties), 1))
    for party_label in target_parties:
        try:
            p_results = collection.query(
                query_embeddings=hyde_embedding,
                n_results=min(per_party_n, n_chunks),
                where={"party": {"$in": [party_label]}},
            )
            for rank in range(len(p_results["ids"][0])):
                doc_id = p_results["ids"][0][rank]
                if doc_id not in party_candidates:
                    party_candidates[doc_id] = {
                        "text":      p_results["documents"][0][rank],
                        "speaker":   p_results["metadatas"][0][rank]["speaker"],
                        "party":     p_results["metadatas"][0][rank]["party"],
                        "date":      p_results["metadatas"][0][rank]["date"],
                        "agenda":    p_results["metadatas"][0][rank]["agenda"],
                        "distance":  p_results["distances"][0][rank],
                        "rrf_party": 1.0 / (60 + rank),
                    }
        except Exception:
            pass

    if target_parties:
        print(f"    [party stream: {target_parties}, {len(party_candidates)} candidates]",
              end=" ", flush=True)

    # stream 5: year-stratified -- embed year-anchored sub-queries for temporal range questions
    year_candidates  = {}
    query_years_list = sorted(set(re.findall(r'\b(201[5-9]|202[0-1])\b', question)))
    if len(query_years_list) >= 2:
        for yr in query_years_list:
            try:
                yr_embedding = model.encode([f"{question} {yr}"]).tolist()
                yr_results = collection.query(query_embeddings=yr_embedding, n_results=5)
                for rank, (doc_id, doc, meta, dist) in enumerate(zip(
                    yr_results["ids"][0], yr_results["documents"][0],
                    yr_results["metadatas"][0], yr_results["distances"][0]
                )):
                    if doc_id not in year_candidates:
                        year_candidates[doc_id] = {
                            "text":     doc,
                            "speaker":  meta["speaker"],
                            "party":    meta["party"],
                            "date":     meta["date"],
                            "agenda":   meta["agenda"],
                            "distance": dist,
                            "rrf_year": 1.0 / (60 + rank),
                        }
            except Exception:
                pass
        if year_candidates:
            print(f"    [year stream: {query_years_list}, {len(year_candidates)} candidates]",
                  end=" ", flush=True)

    # merging all streams using Reciprocal Rank Fusion
    all_ids    = (set(dense_candidates) | set(direct_candidates) |
                  set(bm25_candidates)  | set(party_candidates) | set(year_candidates))
    rrf_scores = {
        doc_id: (
            dense_candidates.get(doc_id,  {}).get("rrf_dense",  0.0) +
            direct_candidates.get(doc_id, {}).get("rrf_direct", 0.0) +
            bm25_candidates.get(doc_id,   {}).get("rrf_bm25",   0.0) +
            party_candidates.get(doc_id,  {}).get("rrf_party",  0.0) +
            year_candidates.get(doc_id,   {}).get("rrf_year",   0.0)
        )
        for doc_id in all_ids
    }

    top_ids    = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:RERANK_POOL]
    candidates = []
    seen_texts = set()
    for doc_id in top_ids:
        chunk = (dense_candidates.get(doc_id) or direct_candidates.get(doc_id)
                 or bm25_candidates.get(doc_id) or party_candidates.get(doc_id)
                 or year_candidates.get(doc_id))
        key = chunk["text"].strip()[:80]
        if key not in seen_texts:
            seen_texts.add(key)
            candidates.append(chunk)

    # reranking the merged candidates with a cross-encoder
    pairs         = [[question, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    # temporal boost: +0.3 for chunks whose year matches a year in the question
    query_years = set(re.findall(r'\b(201[5-9]|202[0-1])\b', question))
    if query_years:
        boosted = []
        for score, chunk in zip(rerank_scores, candidates):
            chunk_year = chunk.get("date", "")[:4]
            bonus = 0.3 if chunk_year in query_years else 0.0
            boosted.append((score + bonus, chunk))
        ranked = sorted(boosted, key=lambda x: -x[0])
    else:
        ranked = sorted(zip(rerank_scores, candidates), key=lambda x: -x[0])

    # party diversity: guarantee at least one chunk per target party in top-k
    if len(target_parties) >= 2:
        per_party_cap = max(1, -(-top_k // len(target_parties)))
        party_counts  = {p: 0 for p in target_parties}
        final         = []
        remaining     = list(ranked)

        for party in target_parties:
            for item in remaining:
                s, c = item
                if c["party"] == party:
                    chunk = dict(c); chunk["rerank_score"] = float(s)
                    final.append(chunk)
                    party_counts[party] = party_counts.get(party, 0) + 1
                    remaining.remove(item)
                    break

        for s, c in remaining:
            if len(final) >= top_k: break
            p = c["party"]
            if p in party_counts and party_counts[p] >= per_party_cap: continue
            chunk = dict(c); chunk["rerank_score"] = float(s)
            final.append(chunk)
            party_counts[p] = party_counts.get(p, 0) + 1

        seen_texts = {c["text"][:80] for c in final}
        for s, c in remaining:
            if len(final) >= top_k: break
            if c["text"][:80] not in seen_texts:
                chunk = dict(c); chunk["rerank_score"] = float(s)
                final.append(chunk); seen_texts.add(c["text"][:80])

        final.sort(key=lambda c: c["rerank_score"], reverse=True)
        return final[:top_k]

    elif len(target_parties) == 1:
        target_party = target_parties[0]
        min_target   = max(1, -(-top_k // 2))
        final        = []
        remaining    = list(ranked)

        for item in remaining[:]:
            if len(final) >= min_target: break
            s, c = item
            if c["party"] == target_party:
                chunk = dict(c); chunk["rerank_score"] = float(s)
                final.append(chunk); remaining.remove(item)

        for s, c in remaining:
            if len(final) >= top_k: break
            chunk = dict(c); chunk["rerank_score"] = float(s)
            final.append(chunk)

        final.sort(key=lambda c: c["rerank_score"], reverse=True)
        return final[:top_k]

    final = []
    for score, chunk in ranked[:top_k]:
        chunk = dict(chunk); chunk["rerank_score"] = float(score)
        final.append(chunk)
    return final


# =============================================================================
# ANSWER GENERATION
# =============================================================================

# build a prompt from retrieved chunks and generate a final answer
def generate(question, context_chunks):
    context = ""
    for i, chunk in enumerate(context_chunks):
        context += (f"\n--- Speech {i+1} ({chunk['speaker']}, {chunk['party']}, "
                    f"{chunk['date']}) ---\n")
        context += chunk["text"] + "\n"

    # tailoring the task instruction based on question type
    q = question.lower()
    if any(w in q for w in ["differ", "compare", "contrast", "versus", " vs ", "between"]):
        task_instruction = (
            "This is a COMPARISON question. Present each party's or speaker's position "
            "as a clearly labelled separate paragraph before drawing any contrast."
        )
    elif any(w in q for w in ["evolve", "evolution", "change", "over time", "from ", "to 20"]):
        task_instruction = (
            "This is a TEMPORAL EVOLUTION question. Organise your answer chronologically, "
            "noting how positions or debates shifted across the years covered by the speeches."
        )
    elif any(w in q for w in ["why", "reason", "cause", "led to", "result"]):
        task_instruction = (
            "This is a CAUSAL question. Identify specific reasons, events, or arguments "
            "from the speeches that explain the outcome."
        )
    else:
        task_instruction = (
            "Answer comprehensively, citing specific speakers by name, their party, "
            "and dates from the speeches provided."
        )

    prompt = f"""You are a research assistant analysing UK parliamentary speeches about Brexit.

Answer the following question based ONLY on the parliamentary speeches provided below.
You MUST cite specific speaker names, party affiliations, and dates from the speeches.
{task_instruction}
If the speeches don't contain enough information to fully answer the question, say so.

SPEECHES:
{context}

QUESTION: {question}

ANSWER:"""

    for attempt in range(3):
        try:
            response = llm_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content or "ERROR: No response from LLM"
        except Exception as e:
            if attempt < 2:
                print(f"  [generate retry {attempt+1}: {type(e).__name__}]")
                time.sleep(5)
            else:
                return "ERROR: LLM unavailable after retries"


# =============================================================================
# PRE-GENERATING HYDE DOCUMENTS
# =============================================================================

# generating HyDE docs before BM25 loads — both together would spike RAM
print(f"\n  Pre-generating HyDE documents for {len(TEST_QUESTIONS)} questions...")
hyde_docs = []
for i, question in enumerate(TEST_QUESTIONS):
    print(f"    Q{i+1}/{len(TEST_QUESTIONS)}: {question[:65]}...", end=" ", flush=True)
    t = time.time()
    hyde_docs.append(generate_hyde(question))
    print(f"({time.time() - t:.1f}s)")
print(f"  HyDE generation complete.")


# =============================================================================
# BUILDING BM25 INDEX
# =============================================================================

print(f"\nBuilding BM25 index over {n_chunks:,} chunks...")
print("-" * 60)

t0 = time.time()
print("  Loading chunks from ChromaDB (batched)...", end=" ", flush=True)

# fetching all chunks in batches to build a full-corpus BM25 index
BATCH_SIZE = 5000
ALL_IDS, ALL_DOCS, ALL_METAS = [], [], []
offset = 0
while offset < n_chunks:
    batch = collection.get(include=["documents", "metadatas"], limit=BATCH_SIZE, offset=offset)
    ALL_IDS   += batch["ids"]
    ALL_DOCS  += batch["documents"]
    ALL_METAS += batch["metadatas"]
    offset += BATCH_SIZE

ID_TO_IDX = {doc_id: i for i, doc_id in enumerate(ALL_IDS)}

# tokenising with stopword removal before fitting BM25
tokenised_corpus = [
    [w for w in doc.lower().split() if w not in _STOP_WORDS]
    for doc in ALL_DOCS
]
bm25 = BM25Okapi(tokenised_corpus)
print(f"done ({time.time() - t0:.1f}s)")

# drop raw text -- BM25 keeps the IDF weights we actually need
del tokenised_corpus, ALL_DOCS
gc.collect()
print("  Index built. Raw text freed.")


# =============================================================================
# RUNNING QUESTIONS
# =============================================================================

print(f"\nRunning {len(TEST_QUESTIONS)} questions through Naive RAG...")
print("=" * 60)

all_retrieve_results = []

for i, (question, hyde_doc) in enumerate(zip(TEST_QUESTIONS, hyde_docs)):
    print(f"\n  Q{i+1}/{len(TEST_QUESTIONS)}: {question}")
    start_time = time.time()
    print("    Retrieving (pre-generated HyDE)...", end=" ", flush=True)
    retrieved_chunks = retrieve(question, hyde_doc=hyde_doc)
    retrieve_time    = time.time() - start_time
    print(f"retrieved {len(retrieved_chunks)} chunks in {retrieve_time:.2f}s")

    for j, chunk in enumerate(retrieved_chunks):
        print(f"    Chunk {j+1}: {chunk['speaker']} ({chunk['party']}, {chunk['date']}) "
              f"rerank={chunk.get('rerank_score', 0):.3f}")

    all_retrieve_results.append((question, retrieved_chunks, retrieve_time))

# freeing heavy structures so Ollama can load cleanly
print("\n  Freeing BM25 and cross-encoder before answer generation...")
del bm25, reranker, ALL_METAS, ALL_IDS, ID_TO_IDX
gc.collect()
print("  Memory freed.")


# =============================================================================
# GENERATING ANSWERS
# =============================================================================

print("\n  Generating answers...")
all_results = []

for i, (question, retrieved_chunks, retrieve_time) in enumerate(all_retrieve_results):
    print(f"\n  Q{i+1}/{len(TEST_QUESTIONS)}: {question[:70]}")
    gen_start = time.time()
    answer    = generate(question, retrieved_chunks)
    gen_time  = time.time() - gen_start
    print(f"    Generated in {gen_time:.1f}s")
    print(f"    Answer: {answer[:200]}...")

    all_results.append({
        "question_id":           QUESTION_IDS[i],
        "question":              question,
        "answer":                answer,
        "retrieved_chunks":      retrieved_chunks,
        "retrieve_time_seconds": round(retrieve_time, 3),
        "generate_time_seconds": round(gen_time, 3),
        "model":                 OLLAMA_MODEL,
        "provider":              PROVIDER,
        "top_k":                 TOP_K,
        "system":                "naive_rag",
        "enhancements":          [
            "party_aware_hyde", "quad_rrf", "global_bm25",
            "party_targeted_stream_hyde", "cross_encoder_rerank",
            "party_diversity_enforcement", "bm25_stopword_filter",
            "implicit_party_inference", "temporal_rerank_boost",
        ],
    })


# =============================================================================
# SAVING RESULTS
# =============================================================================

print(f"\n{'=' * 60}")
print("Saving results...")
print(f"{'=' * 60}")

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"  + {RESULTS_FILE}  ({len(all_results)} results)")

total_retrieve = sum(r["retrieve_time_seconds"] for r in all_results)
total_generate = sum(r["generate_time_seconds"] for r in all_results)
print(f"  Total retrieval time:  {total_retrieve:.1f}s")
print(f"  Total generation time: {total_generate:.1f}s")
print(f"  Average per question:  {(total_retrieve + total_generate) / len(all_results):.1f}s")

# saving a flat CSV for quick inspection in a spreadsheet
csv_cols = [
    "question_id", "question", "answer",
    "retrieve_time_seconds", "generate_time_seconds",
    "chunk1_speaker", "chunk1_party", "chunk1_date", "chunk1_rerank",
    "chunk2_speaker", "chunk2_party", "chunk2_date", "chunk2_rerank",
    "chunk3_speaker", "chunk3_party", "chunk3_date", "chunk3_rerank",
    "chunk4_speaker", "chunk4_party", "chunk4_date", "chunk4_rerank",
    "chunk5_speaker", "chunk5_party", "chunk5_date", "chunk5_rerank",
]

with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_cols)
    writer.writeheader()
    for r in all_results:
        row = {
            "question_id": r["question_id"], "question": r["question"],
            "answer": r["answer"],
            "retrieve_time_seconds": r["retrieve_time_seconds"],
            "generate_time_seconds": r["generate_time_seconds"],
        }
        for j in range(5):
            prefix = f"chunk{j+1}_"
            if j < len(r["retrieved_chunks"]):
                c = r["retrieved_chunks"][j]
                row[prefix + "speaker"] = c.get("speaker", "")
                row[prefix + "party"]   = c.get("party", "")
                row[prefix + "date"]    = c.get("date", "")
                row[prefix + "rerank"]  = round(c.get("rerank_score", 0), 4)
            else:
                row[prefix + "speaker"] = row[prefix + "party"] = ""
                row[prefix + "date"]    = row[prefix + "rerank"] = ""
        writer.writerow(row)

print(f"  + {RESULTS_CSV}")