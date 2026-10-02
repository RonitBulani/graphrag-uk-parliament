# Evaluate Naive RAG vs GraphRAG across 7 metrics aligned to marking criteria.
# Primary evaluator: phi3:mini (more consistent than qwen2.5:3b for RAGAS).

import json
import csv
import re
import os
import asyncio
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision
from ragas.embeddings import BaseRagasEmbedding
from sentence_transformers import SentenceTransformer, CrossEncoder
from bert_score import score as bert_score_fn
from scipy.stats import wilcoxon


# =============================================================================
# CONFIGURATION
# =============================================================================

_ROOT           = Path(__file__).resolve().parent.parent
NAIVE_FILE      = _ROOT / "results" / "naive_rag_results.json"
GRAPH_FILE      = _ROOT / "results" / "graphrag_results.json"
GT_FILE         = _ROOT / "results" / "pooled_references.json"
QUESTIONS_FILE  = _ROOT / "data" / "questions.json"
OUTPUT_JSON     = _ROOT / "results" / "evaluation_results.json"
OUTPUT_CSV      = _ROOT / "results" / "evaluation_results.csv"
FIGURES_DIR     = _ROOT / "figures"

# plotting colors
C_NAIVE = "#4878CF"
C_GRAPH = "#D65F5F"

os.makedirs(FIGURES_DIR, exist_ok=True)

# loading questions to build hop_groups dynamically
with open(QUESTIONS_FILE, encoding="utf-8") as _qf:
    _q_list = json.load(_qf)
HOP_GROUPS: dict[str, list[int]] = {}
for _q in _q_list:
    HOP_GROUPS.setdefault(_q["hop"], []).append(_q["id"])

# loading auto-generated pooled references (parties, speakers, reference answers)
if GT_FILE.exists():
    with open(GT_FILE, encoding="utf-8") as _gtf:
        _gt_raw = json.load(_gtf)
    TARGET_ENTITIES   = {int(k): {"parties": v["target_parties"],
                                   "speakers": v["target_speakers"]}
                         for k, v in _gt_raw.items()}
    REFERENCE_ANSWERS = {int(k): v["reference_answer"] for k, v in _gt_raw.items()}
else:
    # fallback: original q1-q10 hand-written ground truth if file missing
    print("WARNING: results/pooled_references.json not found — using fallback Q1-10 annotations")
    TARGET_ENTITIES = {
        1:  {"parties": ["Con"],               "speakers": ["Theresa May"]},
        2:  {"parties": ["Con"],               "speakers": []},
        3:  {"parties": ["Lab", "Con"],        "speakers": []},
        4:  {"parties": ["Con"],               "speakers": ["Theresa May"]},
        5:  {"parties": ["Con"],               "speakers": ["Theresa May"]},
        6:  {"parties": ["Lab"],               "speakers": ["Keir Starmer"]},
        7:  {"parties": ["Lab", "Con", "SNP"], "speakers": []},
        8:  {"parties": ["SNP", "Con"],        "speakers": []},
        9:  {"parties": ["Con"],               "speakers": []},
        10: {"parties": ["Con"],               "speakers": []},
    }
    REFERENCE_ANSWERS = {
        1: ("Article 50 of the Treaty on European Union is the legal mechanism that "
            "allows a member state to leave the EU. Theresa May formally triggered "
            "Article 50 on 29 March 2017."),
        2: ("The main arguments for leaving the EU centred on sovereignty, immigration "
            "control, and economic self-determination."),
        3: ("The Conservative government sought to end free movement and leave the Single "
            "Market. Labour's position evolved toward supporting a permanent customs union."),
        4: ("The Northern Ireland backstop was a legal guarantee to avoid a hard border. "
            "The DUP and Eurosceptics argued it could trap the UK in EU rules indefinitely."),
        5: ("May's deal included a transition period and the backstop. No deal meant "
            "immediately defaulting to WTO tariffs with no transition."),
        6: ("By 2018 Labour backed a permanent customs union. By 2019 it was a "
            "cross-party flashpoint, with the Commons passing indicative votes for it."),
        7: ("Labour MPs warned of NHS labour shortages. SNP argued Scotland's economy "
            "depended on EU workers. Conservatives said a points-based system would work."),
        8: ("The SNP argued Brexit was imposed on Scotland which voted 62% Remain, "
            "using it to argue for independence and re-joining the EU."),
        9: ("Chequers proposed a common rulebook for goods. Johnson and Davis resigned. "
            "The EU rejected the facilitated customs arrangement."),
        10: ("Conservative MPs voted May's deal down three times. The ERG led by "
             "Rees-Mogg opposed it. Some Remainers also voted against, preferring a "
             "second referendum."),
    }


# =============================================================================
# EVALUATOR SETUP
# =============================================================================

print("Setting up RAGAS evaluators...")

# wrapping sentencetransformer as a ragas-compatible embedding provider
class STEmbedding(BaseRagasEmbedding):
    def __init__(self):
        self._m = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_text(self, text, **kw):
        return self._m.encode(text).tolist()
    def embed_texts(self, texts, **kw):
        return self._m.encode(texts).tolist()
    async def aembed_text(self, text, **kw):
        return self.embed_text(text)
    async def aembed_texts(self, texts, **kw):
        return self.embed_texts(texts)

aclient = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

try:
    import requests as _req
    _models = [m["name"] for m in _req.get("http://localhost:11434/api/tags", timeout=5).json().get("models", [])]
    _phi3_available  = any("phi3" in m for m in _models)
    _qwen_available  = any("qwen2.5:3b" in m for m in _models)
except Exception:
    _phi3_available = _qwen_available = False

# phi3 is preferred due to stability with ragas json parsing
if _phi3_available:
    ragas_llm_primary = llm_factory("phi3:mini",   client=aclient)
    print("  Primary evaluator:    phi3:mini")
else:
    ragas_llm_primary = llm_factory("qwen2.5:3b",  client=aclient)
    print("  Primary evaluator:    qwen2.5:3b (phi3:mini not found)")

ragas_emb  = STEmbedding()
ar_primary = AnswerRelevancy(llm=ragas_llm_primary, embeddings=ragas_emb)
cp_primary = ContextPrecision(llm=ragas_llm_primary)

print("  Loading NLI faithfulness model (cross-encoder/nli-deberta-v3-small)...", end=" ", flush=True)
_nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
_id2label  = _nli_model.model.config.id2label
_ent_idx   = next(k for k, v in _id2label.items() if "entail" in v.lower())
print(f"done  (entailment label idx={_ent_idx}: {_id2label})")
print("  Ready.\n")


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

# standard reference comparison
def bert_score_pair(hypothesis: str, reference: str) -> float:
    _, _, F1 = bert_score_fn(
        [hypothesis], [reference],
        lang="en", model_type="distilbert-base-uncased", verbose=False,
    )
    return round(float(F1[0]), 3)

# fraction of target parties and speakers present in the top-5
def hit_rate(result):
    qid     = result["question_id"]
    targets = TARGET_ENTITIES.get(qid, {})
    all_t   = targets.get("parties", []) + targets.get("speakers", [])
    if not all_t: return None
    ret_p = {c["party"]   for c in result["retrieved_chunks"]}
    ret_s = {c["speaker"] for c in result["retrieved_chunks"]}
    return round(sum(1 for t in all_t if t in ret_p or t in ret_s) / len(all_t), 3)

# reciprocal rank of the first retrieved chunk matching a target speaker/party
def mrr(result):
    qid     = result["question_id"]
    targets = set(TARGET_ENTITIES.get(qid, {}).get("parties", []) +
                  TARGET_ENTITIES.get(qid, {}).get("speakers", []))
    if not targets: return None
    for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
        if chunk["party"] in targets or chunk["speaker"] in targets:
            return round(1 / rank, 3)
    return 0.0

def party_recall(result):
    qid     = result["question_id"]
    targets = set(TARGET_ENTITIES.get(qid, {}).get("parties", []))
    if not targets: return None
    ret = {c["party"] for c in result["retrieved_chunks"]}
    return round(len(targets & ret) / len(targets), 3)

def avg_distance(result):
    d = [c["distance"] for c in result["retrieved_chunks"]]
    return round(sum(d) / len(d), 4)

def party_div_ret(result):
    return len({c["party"] for c in result["retrieved_chunks"]})

def speaker_div_ret(result):
    return len({c["speaker"] for c in result["retrieved_chunks"]})

def speaker_citations(result):
    answer    = result["answer"].lower()
    retrieved = {c["speaker"] for c in result["retrieved_chunks"]}
    return len({s for s in retrieved if s.lower() in answer})

# rank-weighted entity-target relevance over the top 5 retrieved chunks
def ndcg_at_5(result):
    qid     = result["question_id"]
    targets = set(TARGET_ENTITIES.get(qid, {}).get("parties", []) +
                  TARGET_ENTITIES.get(qid, {}).get("speakers", []))
    if not targets:
        return None
    rels = [1 if (c["party"] in targets or c["speaker"] in targets) else 0
            for c in result["retrieved_chunks"][:5]]
    dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(1 / np.log2(i + 2) for i in range(sum(rels)))
    return round(dcg / idcg, 3) if idcg > 0 else 0.0

def precision_at_5(result):
    qid     = result["question_id"]
    targets = set(TARGET_ENTITIES.get(qid, {}).get("parties", []) +
                  TARGET_ENTITIES.get(qid, {}).get("speakers", []))
    if not targets:
        return None
    chunks = result["retrieved_chunks"][:5]
    return round(sum(1 for c in chunks if c["party"] in targets or c["speaker"] in targets) / 5, 3)

# fraction of answer sentences entailed by the retrieved context (hallucination check)
def faithfulness(result):
    chunks    = result["retrieved_chunks"][:5]
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', result["answer"])
                 if len(s.strip()) > 25][:12]
    if not sentences or not chunks:
        return None
    try:
        entailed = 0
        for sent in sentences:
            max_ent = 0.0
            for chunk in chunks:
                ctx   = chunk["text"][:400]
                score = _nli_model.predict([[ctx, sent]], apply_softmax=True)[0]
                max_ent = max(max_ent, float(score[_ent_idx]))
            if max_ent > 0.25:   # low threshold due to cross-domain political text
                entailed += 1
        return round(entailed / len(sentences), 3)
    except Exception:
        return None

# run ragas metrics asynchronously with a fallback retry
async def ragas_scores(result, ar_m, cp_m, truncate=False):
    q   = result["question"]
    a   = result["answer"]
    qid = result["question_id"]

    max_len = 300 if truncate else None
    ctx = [c["text"][:max_len] if max_len else c["text"] for c in result["retrieved_chunks"]]
    ref = " ".join((c["text"][:max_len] if max_len else c["text"]) for c in result["retrieved_chunks"][:2])

    ar_score = None
    for attempt in range(2):
        try:
            ar       = await ar_m.ascore(user_input=q, response=a)
            ar_score = round(ar.value, 3)
            break
        except Exception as e:
            if attempt == 0: await asyncio.sleep(2)
            else: print(f"    RAGAS AR error Q{qid}: {type(e).__name__}")

    cp_score = None
    for attempt in range(2):
        try:
            cp       = await cp_m.ascore(user_input=q, reference=ref, retrieved_contexts=ctx)
            cp_score = round(cp.value, 3)
            break
        except Exception as e:
            if attempt == 0: await asyncio.sleep(2)
            else: print(f"    RAGAS CP error Q{qid}: {type(e).__name__}")

    return ar_score, cp_score


# =============================================================================
# LOADING RESULTS
# =============================================================================

with open(NAIVE_FILE, encoding="utf-8") as f:
    naive_results = json.load(f)
with open(GRAPH_FILE, encoding="utf-8") as f:
    graph_results = json.load(f)

print(f"Loaded: {len(naive_results)} Naive RAG, {len(graph_results)} GraphRAG results\n")


# =============================================================================
# COMPUTING METRICS
# =============================================================================

print("Computing retrieval + answer metrics...")

all_qids = sorted([q["id"] for q in _q_list])
all_scores = {}

for qid in all_qids:
    n   = next(r for r in naive_results if r["question_id"] == qid)
    g   = next(r for r in graph_results if r["question_id"] == qid)
    print(f"  Q{qid} faithfulness...", end=" ", flush=True)
    n_faith = faithfulness(n)
    g_faith = faithfulness(g)
    print(f"N={n_faith}  G={g_faith}")
    all_scores[qid] = {
        "question":  n["question"],
        "hop_group": next(k for k, v in HOP_GROUPS.items() if qid in v),
        "naive": {
            "mrr":              mrr(n),
            "ndcg_5":           ndcg_at_5(n),
            "faithfulness":     n_faith,
            "speaker_div_ret":  speaker_div_ret(n),
            "bert_score":       None,
            "answer_relevancy": None,
            "context_precision":None,
        },
        "graphrag": {
            "mrr":              mrr(g),
            "ndcg_5":           ndcg_at_5(g),
            "faithfulness":     g_faith,
            "speaker_div_ret":  speaker_div_ret(g),
            "bert_score":       None,
            "answer_relevancy": None,
            "context_precision":None,
        },
    }

print("  Non-RAGAS + faithfulness metrics done.")
_primary_name = "phi3:mini" if _phi3_available else "qwen2.5:3b"
print(f"Computing RAGAS scores - primary evaluator: {_primary_name} (~3-5 min)...")

async def run_ragas():
    # running ragas answer relevancy and context precision
    for qid in all_qids:
        n = next(r for r in naive_results if r["question_id"] == qid)
        g = next(r for r in graph_results if r["question_id"] == qid)
        print(f"  Q{qid} [primary]...", end=" ", flush=True)
        n_ar, n_cp = await ragas_scores(n, ar_primary, cp_primary)
        g_ar, g_cp = await ragas_scores(g, ar_primary, cp_primary)
        all_scores[qid]["naive"]["answer_relevancy"]    = n_ar
        all_scores[qid]["naive"]["context_precision"]   = n_cp
        all_scores[qid]["graphrag"]["answer_relevancy"] = g_ar
        all_scores[qid]["graphrag"]["context_precision"]= g_cp
        print("done")

    print()
    print("Computing BERTScore (TREC-pooled reference answers)...")
    for qid in all_qids:
        n   = next(r for r in naive_results if r["question_id"] == qid)
        g   = next(r for r in graph_results if r["question_id"] == qid)
        ref = REFERENCE_ANSWERS.get(qid, "")
        if not ref:
            print(f"  Q{qid}... skipped (no reference)")
            continue
        print(f"  Q{qid}...", end=" ", flush=True)
        n_bs = bert_score_pair(n["answer"], ref)
        g_bs = bert_score_pair(g["answer"], ref)
        all_scores[qid]["naive"]["bert_score"]    = n_bs
        all_scores[qid]["graphrag"]["bert_score"] = g_bs
        print(f"done  (N={n_bs:.3f}  G={g_bs:.3f})")

asyncio.run(run_ragas())
print()


# =============================================================================
# PRINTING RESULTS TABLES
# =============================================================================

def fmt(v, f=".3f"):
    return f"{v:{f}}" if v is not None else "   N/A"

def safe_avg(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None

all_cols = [
    ("mrr",              "MRR",        False),
    ("ndcg_5",           "nDCG@5",     False),
    ("bert_score",       "BERTScore",  False),
    ("faithfulness",     "Faithful",   False),
    ("answer_relevancy", "AnsRel",     False),
    ("context_precision","CtxPrec",    False),
    ("speaker_div_ret",  "SpkDiv",     False),
]

print("=" * 100)
print("ALL METRICS  (N = Naive, G = GraphRAG)")
header = f"{'Q':>3}  {'Hop':>5}"
for _, lbl, _ in all_cols:
    header += f"  {'N-'+lbl:>10} {'G-'+lbl:>10}"
print(header)
print("-" * 100)
for qid in all_qids:
    s   = all_scores[qid]
    row = f"{qid:>3}  {s['hop_group']:>5}"
    for key, _, _ in all_cols:
        nv = s["naive"][key]; gv = s["graphrag"][key]
        row += f"  {fmt(nv):>10} {fmt(gv):>10}"
    print(row)

print("\n" + "=" * 75)
print("OVERALL SUMMARY")
print(f"{'Metric':<18}  {'Naive':>8}  {'GraphRAG':>8}  {'Delta':>8}  Winner")
print("-" * 55)
for key, label, lower_better in all_cols:
    n_v = safe_avg([all_scores[q]["naive"][key]    for q in all_qids])
    g_v = safe_avg([all_scores[q]["graphrag"][key] for q in all_qids])
    if n_v is None or g_v is None: continue
    delta  = g_v - n_v
    winner = ("N/A" if lower_better is None else
              "GraphRAG" if (lower_better and delta < 0) or (not lower_better and delta > 0)
              else "Naive")
    print(f"{label:<18}  {n_v:>8.3f}  {g_v:>8.3f}  {delta:>+8.3f}  {winner}")

print("\nBY HOP GROUP")
print(f"{'Hop':>5}  {'Metric':<18}  {'Naive':>8}  {'GraphRAG':>8}  {'Delta':>8}  Winner")
print("-" * 65)
for hop, qids in HOP_GROUPS.items():
    for key, label, lower_better in all_cols:
        n_v = safe_avg([all_scores[q]["naive"][key]    for q in qids])
        g_v = safe_avg([all_scores[q]["graphrag"][key] for q in qids])
        if n_v is None or g_v is None: continue
        delta  = g_v - n_v
        winner = ("N/A" if lower_better is None else
                  "GraphRAG" if (lower_better and delta < 0) or (not lower_better and delta > 0)
                  else "Naive")
        print(f"{hop:>5}  {label:<18}  {n_v:>8.3f}  {g_v:>8.3f}  {delta:>+8.3f}  {winner}")
    print()

print("\n" + "=" * 75)
print("WILCOXON SIGNED-RANK TESTS  (GraphRAG vs Naive RAG, two-sided, α=0.05)")
print(f"  N = {len(all_qids)} paired questions")
print(f"{'Metric':<18}  {'N-mean':>8}  {'G-mean':>8}  {'stat':>8}  {'p-value':>10}  Sig?")
print("-" * 65)
for key, label, _ in all_cols:
    pairs = [(all_scores[q]["naive"][key], all_scores[q]["graphrag"][key])
             for q in all_qids
             if all_scores[q]["naive"][key] is not None
             and all_scores[q]["graphrag"][key] is not None]
    if len(pairs) < 5:
        print(f"{label:<18}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {'N/A':>10}  —")
        continue
    n_vals = [p[0] for p in pairs]
    g_vals = [p[1] for p in pairs]
    diffs  = [g - n for n, g in pairs]
    if all(d == 0 for d in diffs):
        print(f"{label:<18}  {sum(n_vals)/len(n_vals):>8.3f}  {sum(g_vals)/len(g_vals):>8.3f}"
              f"  {'N/A':>8}  {'N/A':>10}  (tied)")
        continue
    try:
        stat, pval = wilcoxon(g_vals, n_vals, alternative="two-sided")
        sig = "YES *" if pval < 0.05 else "no"
        print(f"{label:<18}  {sum(n_vals)/len(n_vals):>8.3f}  {sum(g_vals)/len(g_vals):>8.3f}"
              f"  {stat:>8.1f}  {pval:>10.4f}  {sig}")
    except Exception as e:
        print(f"{label:<18}  error: {e}")
print()


# =============================================================================
# GENERATING FIGURES
# =============================================================================

def hop_avg(metric, system):
    return [
        safe_avg([all_scores[q][system][metric] for q in HOP_GROUPS[h]])
        for h in ["1-hop", "2-hop", "3-hop"]
    ]

hop_labels = [
    f"{h}\n(Q{min(qids)}-{max(qids)})" if len(qids) > 1 else f"{h}\n(Q{qids[0]})"
    for h, qids in HOP_GROUPS.items()
]
x     = np.arange(len(HOP_GROUPS))
width = 0.35

# draw side-by-side bars for naive vs graphrag
def annotated_bars(ax, x, n_vals, g_vals, ylim=1.2):
    n_vals = [v if v is not None else 0 for v in n_vals]
    g_vals = [v if v is not None else 0 for v in g_vals]
    bn = ax.bar(x - width/2, n_vals, width, label="Naive RAG", color=C_NAIVE, alpha=0.85)
    bg = ax.bar(x + width/2, g_vals, width, label="GraphRAG",  color=C_GRAPH, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(hop_labels)
    if ylim: ax.set_ylim(0, ylim)
    ax.legend(fontsize=8)
    for bar in list(bn) + list(bg):
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)

# fig 1: retrieval metrics
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (metric, title) in zip(axes, [
    ("mrr",    "MRR (first target-entity chunk)"),
    ("ndcg_5", "nDCG@5 (target-entity relevance)"),
]):
    annotated_bars(ax, x, hop_avg(metric, "naive"), hop_avg(metric, "graphrag"))
    ax.set_title(title); ax.set_ylabel("Score (0-1)")
plt.suptitle("Retrieval Quality by Hop Count", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig1_retrieval.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig1_retrieval.png")

# fig 2: answer quality metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (metric, title) in zip(axes, [
    ("bert_score",   "BERTScore F1 (vs TREC reference)"),
    ("faithfulness", "NLI Faithfulness (% sentences entailed)"),
    ("speaker_div_ret", "Speaker Diversity (unique MPs retrieved)"),
]):
    annotated_bars(ax, x, hop_avg(metric, "naive"), hop_avg(metric, "graphrag"))
    ax.set_title(title)
plt.suptitle("Answer Quality by Hop Count", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig2_answer_quality.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig2_answer_quality.png")

# fig 3: ragas metrics
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (metric, title) in zip(axes, [
    ("answer_relevancy",  "Answer Relevancy (RAGAS)"),
    ("context_precision", "Context Precision (RAGAS)"),
]):
    annotated_bars(ax, x, hop_avg(metric, "naive"), hop_avg(metric, "graphrag"))
    ax.set_title(title); ax.set_ylabel("Score (0-1)")
plt.suptitle("LLM-as-Judge Metrics by Hop Count", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig3_ragas.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig3_ragas.png")

# fig 4: normalised radar plot
def sys_avg(m, sys):
    v = [all_scores[q][sys][m] for q in all_qids if all_scores[q][sys][m] is not None]
    return sum(v)/len(v) if v else 0

radar_labels = ["MRR", "nDCG@5", "BERTScore", "Faithful", "AnsRel", "CtxPrec", "SpkDiv"]
radar_keys   = ["mrr", "ndcg_5", "bert_score", "faithfulness",
                "answer_relevancy", "context_precision", "speaker_div_ret"]
raw_n = [sys_avg(m, "naive")    for m in radar_keys]
raw_g = [sys_avg(m, "graphrag") for m in radar_keys]

norm_n, norm_g = [], []
for nv, gv in zip(raw_n, raw_g):
    lo, hi = min(nv, gv), max(nv, gv)
    span   = (hi - lo) or 1
    norm_n.append((nv - lo) / span)
    norm_g.append((gv - lo) / span)

N        = len(radar_labels)
angles   = [i / N * 2 * np.pi for i in range(N)] + [0]
norm_n_p = norm_n + [norm_n[0]]
norm_g_p = norm_g + [norm_g[0]]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, norm_n_p, "o-", lw=2, color=C_NAIVE, label="Naive RAG")
ax.fill(angles, norm_n_p, alpha=0.15, color=C_NAIVE)
ax.plot(angles, norm_g_p, "o-", lw=2, color=C_GRAPH, label="GraphRAG")
ax.fill(angles, norm_g_p, alpha=0.15, color=C_GRAPH)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_labels, size=9)
ax.set_ylim(0, 1)
ax.set_title("Overall System Profile\n(normalised, outer = better)", size=11, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
fig.savefig(os.path.join(FIGURES_DIR, "fig4_radar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig4_radar.png")


# =============================================================================
# SAVING RESULTS
# =============================================================================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_scores, f, indent=2, ensure_ascii=False)
print(f"\nSaved full evaluation to: {OUTPUT_JSON}")

csv_cols = [
    "question_id", "hop_group", "question", "system",
    "mrr", "ndcg_5", "bert_score", "faithfulness",
    "answer_relevancy", "context_precision", "speaker_div_ret",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
    writer.writeheader()
    for qid in all_qids:
        s = all_scores[qid]
        for system in ("naive", "graphrag"):
            row = {"question_id": qid, "hop_group": s["hop_group"],
                   "question": s["question"], "system": system}
            row.update(s[system])
            writer.writerow(row)

print(f"Saved CSV to: {OUTPUT_CSV}")