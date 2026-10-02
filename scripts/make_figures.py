"""
Generates 7 publication-quality figures for the SCC453 paper.
Run from project root: python paper/make_figures.py

Outputs:
  paper/fig1_hop_comparison.png  — MRR + nDCG@5 by hop group
  paper/fig2_answer_quality.png  — BERTScore + Faithfulness + SpkDiv by hop
  paper/fig3_ragas.png           — RAGAS AnsRel + CtxPrec by hop
  paper/fig4_radar.png           — Normalised radar over all 7 metrics
  paper/fig5_faithfulness.png    — NLI faithfulness per question
  paper/fig6_distributions.png   — Box plots of MRR/nDCG/SpkDiv with Wilcoxon p-values
  paper/fig7_mrr_delta.png       — Per-question MRR delta coloured by hop group
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats


# ==============================
# Loading Data
# ==============================

_ROOT = Path(__file__).resolve().parent.parent
with open(_ROOT / "results" / "evaluation_results.json", encoding="utf-8") as f:
    ev = json.load(f)
with open(_ROOT / "data" / "questions.json", encoding="utf-8") as f:
    _qs = json.load(f)

OUT = _ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_NAIVE = "#4878CF"
C_GRAPH = "#D65F5F"

# building hop groups from questions.json (same logic as evaluation script)
HOP_GROUPS: dict[str, list[int]] = {}
for q in _qs:
    HOP_GROUPS.setdefault(q["hop"], []).append(q["id"])
all_qids = sorted(int(k) for k in ev.keys())

HOPS    = ["1-hop", "2-hop", "3-hop"]
hop_n   = {h: len(HOP_GROUPS[h]) for h in HOPS}
hop_lbls = [f"{h}\n(n={hop_n[h]})" for h in HOPS]
x       = np.arange(3)
width   = 0.32


# ==============================
# Shared Helpers
# ==============================

def safe_avg(vals):
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def hop_avg(metric, system):
    return [safe_avg([ev[str(q)][system][metric] for q in HOP_GROUPS[h]]) for h in HOPS]


def annotated_bars(ax, n_vals, g_vals, ylim=1.15):
    """Draw side-by-side bars with value labels and highlighted 3-hop bars."""
    n_vals = [v if v is not None else 0 for v in n_vals]
    g_vals = [v if v is not None else 0 for v in g_vals]
    bn = ax.bar(x - width/2, n_vals, width, color=C_NAIVE, alpha=0.88,
                label="Naive RAG", edgecolor="white", linewidth=0.5)
    bg = ax.bar(x + width/2, g_vals, width, color=C_GRAPH, alpha=0.88,
                label="GraphRAG", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(hop_lbls, fontsize=8.5)
    ax.set_ylim(0, ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for bar in list(bn) + list(bg):
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.022,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    # highlighting 3-hop bars to draw attention to the key finding
    for b in [bn[2], bg[2]]:
        b.set_edgecolor("#333"); b.set_linewidth(1.5)
    return bn, bg


# shared legend for all bar-chart figures
_legend_handles = [mpatches.Patch(color=C_NAIVE, label="Naive RAG"),
                   mpatches.Patch(color=C_GRAPH, label="GraphRAG")]


# ==============================
# Figure 1 — Retrieval Quality by Hop
# ==============================

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Retrieval Quality by Reasoning Complexity  (N=50 questions)",
             fontsize=13, fontweight="bold", y=1.02)

for ax, (metric, title, ylabel) in zip(axes, [
    ("mrr",    "MRR  (first target-entity chunk)",    "Score (0–1)"),
    ("ndcg_5", "nDCG@5  (target-entity relevance)",      "Score (0–1)"),
]):
    annotated_bars(ax, hop_avg(metric, "naive"), hop_avg(metric, "graphrag"))
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=9)

fig.legend(handles=_legend_handles, loc="upper center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 1.07), frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_hop_comparison.png"), dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig1_hop_comparison.png")


# ==============================
# Figure 2 — Answer Quality by Hop
# ==============================

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Answer Quality by Reasoning Complexity  (N=50 questions)",
             fontsize=13, fontweight="bold", y=1.02)

metrics2 = [
    ("bert_score",      "BERTScore F1  (vs TREC reference)",          "Score (0–1)"),
    ("faithfulness",    "NLI Faithfulness  (% sentences entailed)",   "Score (0–1)"),
    ("speaker_div_ret", "Speaker Diversity  (unique MPs retrieved)",  "Count"),
]
for ax, (metric, title, ylabel) in zip(axes, metrics2):
    ylim = 7.0 if metric == "speaker_div_ret" else 1.15
    annotated_bars(ax, hop_avg(metric, "naive"), hop_avg(metric, "graphrag"), ylim=ylim)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=9)

fig.legend(handles=_legend_handles, loc="upper center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 1.07), frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_answer_quality.png"), dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig2_answer_quality.png")


# ==============================
# Figure 3 — LLM-as-Judge by Hop
# ==============================

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("LLM-as-Judge Metrics by Reasoning Complexity  (N=50 questions)",
             fontsize=13, fontweight="bold", y=1.02)

for ax, (metric, title) in zip(axes, [
    ("answer_relevancy",  "Answer Relevancy  (RAGAS / phi3:mini)"),
    ("context_precision", "Context Precision  (RAGAS / phi3:mini)"),
]):
    annotated_bars(ax, hop_avg(metric, "naive"), hop_avg(metric, "graphrag"))
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel("Score (0–1)", fontsize=9)

fig.legend(handles=_legend_handles, loc="upper center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 1.07), frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_ragas.png"), dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig3_ragas.png")


# ==============================
# Figure 4 — Radar: All 7 Metrics
# ==============================

radar_keys   = ["mrr", "ndcg_5", "bert_score", "faithfulness",
                "answer_relevancy", "context_precision", "speaker_div_ret"]
radar_labels = ["MRR", "nDCG@5", "BERTScore", "Faithful",
                "AnsRel", "CtxPrec", "SpkDiv"]

# computing raw averages then normalising per-axis so outer edge = better
raw_n = [safe_avg([ev[str(q)]["naive"][m]    for q in all_qids]) or 0 for m in radar_keys]
raw_g = [safe_avg([ev[str(q)]["graphrag"][m] for q in all_qids]) or 0 for m in radar_keys]

norm_n, norm_g = [], []
for nv, gv in zip(raw_n, raw_g):
    lo, hi = min(nv, gv), max(nv, gv)
    span = (hi - lo) or 1
    norm_n.append((nv - lo) / span)
    norm_g.append((gv - lo) / span)

N        = len(radar_labels)
angles   = [i / N * 2 * np.pi for i in range(N)] + [0]
norm_n_p = norm_n + [norm_n[0]]
norm_g_p = norm_g + [norm_g[0]]

fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
ax.plot(angles, norm_n_p, "o-", lw=2, color=C_NAIVE, label="Naive RAG")
ax.fill(angles, norm_n_p, alpha=0.12, color=C_NAIVE)
ax.plot(angles, norm_g_p, "o-", lw=2, color=C_GRAPH, label="GraphRAG")
ax.fill(angles, norm_g_p, alpha=0.12, color=C_GRAPH)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, size=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7, color="gray")

# annotating raw scores next to each spoke
for i, angle in enumerate(angles[:-1]):
    offset = 0.20
    ax.annotate(f"{raw_n[i]:.2f}", xy=(angle, norm_n[i]),
                xytext=(angle, norm_n[i] + offset),
                fontsize=7, color=C_NAIVE, ha="center", va="center")
    ax.annotate(f"{raw_g[i]:.2f}", xy=(angle, norm_g[i]),
                xytext=(angle, max(norm_g[i] - offset, 0.0)),
                fontsize=7, color=C_GRAPH, ha="center", va="center")

ax.set_title("Overall System Profile  (N=50 questions)\n"
             "Normalised: outer edge = better on that metric",
             size=11, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10, frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_radar.png"), dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig4_radar.png")


# ==============================
# Figure 5 — NLI Faithfulness per Question
# ==============================

hop_colours = {"1-hop": "#a8d5f5", "2-hop": "#ffe0a0", "3-hop": "#f5b8b8"}

n_faith = [ev[str(q)]["naive"]["faithfulness"]    or 0.0 for q in all_qids]
g_faith = [ev[str(q)]["graphrag"]["faithfulness"] or 0.0 for q in all_qids]
x50     = np.arange(len(all_qids))

fig, ax = plt.subplots(figsize=(18, 4.5))
for i, qid in enumerate(all_qids):
    hop = ev[str(qid)]["hop_group"]
    col = hop_colours[hop]
    ax.bar(x50[i] - width/2, n_faith[i], width, color=C_NAIVE, alpha=0.75, zorder=2)
    ax.bar(x50[i] + width/2, g_faith[i], width, color=C_GRAPH, alpha=0.75, zorder=2)
    ax.axvspan(x50[i] - 0.5, x50[i] + 0.5, color=col, alpha=0.18, zorder=1)

# hop group separators
boundaries = {}
for h in HOPS:
    boundaries[h] = (min(HOP_GROUPS[h]) - 1, max(HOP_GROUPS[h]) - 1)

for h in HOPS:
    lo_i, hi_i = boundaries[h]
    mid_i = (lo_i + hi_i) / 2
    ax.text(mid_i, 0.55, h, ha="center", fontsize=9,
            color={"1-hop": "#2060a0", "2-hop": "#a06000", "3-hop": "#a02020"}[h],
            fontweight="bold")

if lo_i > 0:
    ax.axvline(boundaries["1-hop"][1] + 0.5, color="gray", linestyle="--", lw=1, alpha=0.5)
    ax.axvline(boundaries["2-hop"][1] + 0.5, color="gray", linestyle="--", lw=1, alpha=0.5)

# drawing average lines for both systems
avg_n = sum(n_faith) / len(n_faith)
avg_g = sum(g_faith) / len(g_faith)
ax.axhline(avg_n, color=C_NAIVE, linestyle=":", lw=1.5, alpha=0.8,
           label=f"Naive avg ({avg_n:.3f})")
ax.axhline(avg_g, color=C_GRAPH, linestyle=":", lw=1.5, alpha=0.8,
           label=f"GraphRAG avg ({avg_g:.3f})")

ax.set_xticks(x50)
ax.set_xticklabels([f"Q{q}" for q in all_qids], fontsize=6.5, rotation=45)
ax.set_ylabel("NLI Faithfulness", fontsize=9.5)
ax.set_ylim(0, 0.65)
ax.set_title("NLI Faithfulness per Question  (N=50)\n"
             "Fraction of answer sentences entailed by retrieved context",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax.set_axisbelow(True)

sys_handles = [mpatches.Patch(color=C_NAIVE, alpha=0.75, label="Naive RAG"),
               mpatches.Patch(color=C_GRAPH, alpha=0.75, label="GraphRAG")]
l1 = ax.legend(handles=sys_handles, loc="upper left", fontsize=8.5, framealpha=0.8)
ax.add_artist(l1)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_faithfulness.png"), dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig5_faithfulness.png")


# ==============================
# Figure 6 — Box Plots with Wilcoxon p-values
# ==============================

# shows WHY only SpkDiv reaches significance: clear separation vs overlapping distributions

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Metric Distributions Across 50 Questions\n"
             "(box plots support Wilcoxon signed-rank results)",
             fontsize=12, fontweight="bold", y=1.02)

box_metrics = [
    ("mrr",            "MRR",              "Score (0–1)"),
    ("ndcg_5",         "nDCG@5",           "Score (0–1)"),
    ("speaker_div_ret","Speaker Diversity","Unique MPs"),
]

for ax, (metric, title, ylabel) in zip(axes, box_metrics):
    n_vals = [ev[str(q)]["naive"][metric]    or 0 for q in all_qids]
    g_vals = [ev[str(q)]["graphrag"][metric] or 0 for q in all_qids]

    bp = ax.boxplot([n_vals, g_vals], patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor(C_NAIVE); bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(C_GRAPH);  bp["boxes"][1].set_alpha(0.75)

    # overlaying individual data points with jitter
    jitter = 0.12
    np.random.seed(42)
    for i, (vals, col) in enumerate([(n_vals, C_NAIVE), (g_vals, C_GRAPH)], 1):
        ax.scatter(np.random.normal(i, jitter, len(vals)), vals,
                   color=col, alpha=0.35, s=18, zorder=3)

    # annotating with Wilcoxon p-value
    stat, p = stats.wilcoxon(n_vals, g_vals)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    ymax = max(max(n_vals), max(g_vals)) * 1.08
    ax.annotate(f"p={p:.3f} {sig}", xy=(1.5, ymax),
                ha="center", fontsize=9, color="#333")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Naive RAG", "GraphRAG"], fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_distributions.png"), dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig6_distributions.png")


# ==============================
# Figure 7 — Per-question MRR Delta by Hop
# ==============================

# visualises the hop-complexity gradient at individual question level

hop_colours_bar = {"1-hop": C_NAIVE, "2-hop": "#e09c1a", "3-hop": C_GRAPH}

mrr_delta = [
    ev[str(q)]["graphrag"]["mrr"] - ev[str(q)]["naive"]["mrr"]
    for q in all_qids
]
q_hops = [ev[str(q)]["hop_group"] for q in all_qids]
bar_colors = [hop_colours_bar[h] for h in q_hops]

fig, ax = plt.subplots(figsize=(18, 4.5))
bars = ax.bar(x50, mrr_delta, color=bar_colors, alpha=0.82, edgecolor="white", linewidth=0.4)

ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
avg_delta = sum(mrr_delta) / len(mrr_delta)
ax.axhline(avg_delta, color="#555", linewidth=1.2, linestyle=":",
           label=f"Overall avg delta ({avg_delta:+.3f})")

# drawing per-hop average delta lines
for h, col in hop_colours_bar.items():
    ids = HOP_GROUPS[h]
    hd = [ev[str(q)]["graphrag"]["mrr"] - ev[str(q)]["naive"]["mrr"] for q in ids]
    hm = sum(hd) / len(hd)
    lo = min(all_qids.index(i) for i in ids)
    hi = max(all_qids.index(i) for i in ids)
    ax.plot([lo - 0.4, hi + 0.4], [hm, hm], color=col,
            linewidth=2.2, linestyle="--", alpha=0.9)

# labelling each hop group
for h, col in hop_colours_bar.items():
    ids = HOP_GROUPS[h]
    idxs = [all_qids.index(i) for i in ids]
    mid = (min(idxs) + max(idxs)) / 2
    ax.text(mid, ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 0 else 0.15,
            h, ha="center", fontsize=9, color=col, fontweight="bold")

# adding hop group separators
for h in ["1-hop", "2-hop"]:
    hi_idx = max(all_qids.index(i) for i in HOP_GROUPS[h])
    ax.axvline(hi_idx + 0.5, color="gray", linestyle="--", lw=0.9, alpha=0.5)

ax.set_xticks(x50)
ax.set_xticklabels([f"Q{q}" for q in all_qids], fontsize=6.5, rotation=45)
ax.set_ylabel("MRR Delta (GraphRAG − Naive)", fontsize=9.5)
ax.set_title("Per-Question MRR Delta Coloured by Reasoning Complexity  (N=50)\n"
             "Dashed lines = hop-group average delta; positive = GraphRAG wins",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right", framealpha=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax.set_axisbelow(True)

hop_handles = [mpatches.Patch(color=c, alpha=0.82, label=h)
               for h, c in hop_colours_bar.items()]
l1 = ax.legend(handles=hop_handles, loc="upper left", fontsize=8.5, framealpha=0.8)
ax.add_artist(l1)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_mrr_delta.png"), dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig7_mrr_delta.png")

print(f"\nAll 7 figures saved to: {OUT}")