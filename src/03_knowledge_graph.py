# Build a speaker-party-topic-year knowledge graph from the filtered speeches (runs once)
# Saves knowledge_graph.graphml and knowledge_graph_metadata.json

import pandas as pd
import networkx as nx
import re
import json
from collections import defaultdict, Counter
from textblob import TextBlob
from pathlib import Path


# =============================================================================
# PREPROCESSING HELPERS
# =============================================================================

# strip honorifics so speaker names are consistent across all scripts
_HONORIFIC = re.compile(
    r"^(?:The\s+)?(?:Right\s+Hon(?:ourable)?\.?\s+)?"
    r"(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?|Prof\.?|Sir|Dame|Lord|Lady|Baron(?:ess)?|Rev\.?)\.?\s+",
    re.IGNORECASE,
)
_PROCEDURAL_SPEAKERS = {
    "CHAIR", "THE CHAIR", "DEPUTY SPEAKER", "THE DEPUTY SPEAKER",
    "SPEAKER", "THE SPEAKER", "MADAM DEPUTY SPEAKER", "MR DEPUTY SPEAKER",
    "MADAM SPEAKER", "LORD SPEAKER",
}

# remove honorifics from a speaker name
def normalize_speaker(name):
    if not isinstance(name, str):
        return name
    return _HONORIFIC.sub("", name.strip()).strip()


# =============================================================================
# CONFIGURATION
# =============================================================================

# minimum thresholds for keeping nodes / edges in the graph
MIN_SPEAKER_SPEECHES = 5
MIN_TOPIC_SPEECHES   = 5
MIN_TOPIC_SPEAKERS   = 3
MIN_DEBATED_WITH     = 3

_ROOT      = Path(__file__).resolve().parent.parent
DATA_FILE  = _ROOT / "data" / "parlspeech_brexit_filtered.csv"
GRAPH_FILE = _ROOT / "data" / "knowledge_graph.graphml"
GRAPH_JSON = _ROOT / "data" / "knowledge_graph_metadata.json"
BATCH_SIZE = 5000


# =============================================================================
# LOADING AND FILTERING DATA
# =============================================================================

print("\nLoading data")

df = pd.read_csv(DATA_FILE)
print(f"  {len(df):,} speeches loaded")

# normalise speaker names and drop procedural rows
df["speaker"] = df["speaker"].apply(normalize_speaker)
before = len(df)
df = df[~df["speaker"].isin(_PROCEDURAL_SPEAKERS)]
df = df[df["party"].notna()]
print(f"  Dropped {before - len(df):,} procedural / unknown-party rows")

# keep only speakers with enough speeches to be meaningful graph nodes
speaker_counts = df["speaker"].value_counts()
valid_speakers = speaker_counts[speaker_counts >= MIN_SPEAKER_SPEECHES].index
before = len(df)
df = df[df["speaker"].isin(valid_speakers)]
print(f"  Dropped {before - len(df):,} rows (speakers with < {MIN_SPEAKER_SPEECHES} speeches)")
print(f"  {df['speaker'].nunique()} unique speakers, "
      f"{df['party'].nunique()} parties, {df['agenda'].nunique()} agendas")


# =============================================================================
# CLEANING TOPIC NAMES
# =============================================================================

print("\nCleaning topic names")

# turn a raw agenda string into a short topic label
def clean_topic(agenda):
    if pd.isna(agenda):
        return "Unknown"
    topic = str(agenda)
    # take the top-level category before any ">" separators
    topic = topic.split(">")[0].strip()
    topic = re.sub(r"\[[^\]]*\]", "", topic)
    topic = re.sub(r"\(\s*No\.?\s*\d+\s*\)", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"\s+", " ", topic).strip()
    return topic if len(topic) >= 3 else "Unknown"

df["topic"] = df["agenda"].apply(clean_topic)
df["year"]  = pd.to_datetime(df["date"]).dt.year

print(f"  {df['topic'].nunique()} unique topics")
print(f"  Sample topics:")
for t in df["topic"].value_counts().head(5).index:
    print(f"    - {t}")


# =============================================================================
# COMPUTING SENTIMENT SCORES
# =============================================================================

print("\nCalculating sentiment scores")

# TextBlob polarity -- a lightweight proxy for speaker stance
sentiments = []
for i in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE]
    batch_sentiments = batch["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    sentiments.extend(batch_sentiments.tolist())
    print(f"  {min(i + BATCH_SIZE, len(df)):,}/{len(df):,} speeches")

df["sentiment"] = sentiments
print(f"  Average sentiment: {df['sentiment'].mean():.3f}")
print(f"  Range: {df['sentiment'].min():.3f} to {df['sentiment'].max():.3f}")


# =============================================================================
# BUILDING THE KNOWLEDGE GRAPH
# =============================================================================

print("\nBuilding knowledge graph")

G = nx.Graph()

# party nodes
parties = df["party"].dropna().unique()
for party in parties:
    G.add_node(party, node_type="PARTY")
print(f"  + {len(parties)} party nodes")

# year nodes
years = sorted(df["year"].unique())
for year in years:
    G.add_node(str(year), node_type="YEAR")
print(f"  + {len(years)} year nodes")

# speaker nodes with MEMBER_OF edges to their party
speaker_info = df.groupby("speaker").agg(
    party=("party", "first"),
    speech_count=("text", "count"),
    avg_sentiment=("sentiment", "mean"),
).reset_index()

for _, row in speaker_info.iterrows():
    speaker = row["speaker"]
    G.add_node(speaker, node_type="SPEAKER",
               speech_count=int(row["speech_count"]),
               avg_sentiment=round(row["avg_sentiment"], 3))
    if pd.notna(row["party"]):
        G.add_edge(speaker, row["party"], edge_type="MEMBER_OF")

print(f"  + {len(speaker_info)} speaker nodes with MEMBER_OF edges")

# drop topics that don't meet minimum thresholds
speaker_topic = df.groupby(["speaker", "topic"]).agg(
    count=("text", "count"),
    avg_sentiment=("sentiment", "mean"),
    dates=("date", list),
    years=("year", list),
).reset_index()

topic_stats = speaker_topic.groupby("topic").agg(
    total_speeches=("count", "sum"),
    num_speakers=("speaker", "count"),
).reset_index()
valid_topics = topic_stats[
    (topic_stats["total_speeches"] >= MIN_TOPIC_SPEECHES) &
    (topic_stats["num_speakers"]   >= MIN_TOPIC_SPEAKERS)
]["topic"]
speaker_topic = speaker_topic[speaker_topic["topic"].isin(valid_topics)]
print(f"  Kept {len(valid_topics):,} topics (>= {MIN_TOPIC_SPEECHES} speeches, "
      f">= {MIN_TOPIC_SPEAKERS} speakers) out of {len(topic_stats):,}")

# topic nodes + SPOKE_ABOUT edges with per-year speech counts
topic_nodes_added = set()
for _, row in speaker_topic.iterrows():
    topic = row["topic"]
    if topic not in topic_nodes_added:
        G.add_node(topic, node_type="TOPIC")
        topic_nodes_added.add(topic)

    year_counts = dict(Counter(str(y) for y in row["years"]))
    G.add_edge(row["speaker"], topic,
               edge_type="SPOKE_ABOUT",
               speech_count=int(row["count"]),
               avg_sentiment=round(row["avg_sentiment"], 3),
               year_counts=json.dumps(year_counts))

print(f"  + {len(topic_nodes_added)} topic nodes with SPOKE_ABOUT edges")

# link topics to the years they were debated in
topic_year = df[df["topic"].isin(valid_topics)].groupby(["topic", "year"]).size().reset_index(name="count")
for _, row in topic_year.iterrows():
    G.add_edge(row["topic"], str(row["year"]),
               edge_type="DISCUSSED_IN",
               speech_count=int(row["count"]))
print(f"  + {len(topic_year)} DISCUSSED_IN edges (Topic -> Year)")

# build co-debate edges between speakers appearing in the same debate
print("  Building DEBATED_WITH edges (this may take a moment)")
debate_groups = df.groupby(["date", "agenda"])["speaker"].apply(set).reset_index()
debated_pairs = defaultdict(int)

for _, row in debate_groups.iterrows():
    speakers = list(row["speaker"])
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            pair = tuple(sorted([speakers[i], speakers[j]]))
            debated_pairs[pair] += 1

# keep only pairs that co-debated enough times to be meaningful
strong_pairs = {pair: count for pair, count in debated_pairs.items() if count >= MIN_DEBATED_WITH}
for (s1, s2), count in strong_pairs.items():
    if G.has_node(s1) and G.has_node(s2):
        G.add_edge(s1, s2, edge_type="DEBATED_WITH", debate_count=count)
print(f"  + {len(strong_pairs)} DEBATED_WITH edges")


# =============================================================================
# SAVING THE GRAPH
# =============================================================================

print("\nSaving graph")

nx.write_graphml(G, GRAPH_FILE)
print(f"  + {GRAPH_FILE}")

# save a flat metadata JSON for fast lookup in query scripts
graph_metadata = {
    "speakers": {
        node: dict(G.nodes[node])
        for node in G.nodes if G.nodes[node].get("node_type") == "SPEAKER"
    },
    "parties": [n for n in G.nodes if G.nodes[n].get("node_type") == "PARTY"],
    "topics":  [n for n in G.nodes if G.nodes[n].get("node_type") == "TOPIC"],
    "years":   [n for n in G.nodes if G.nodes[n].get("node_type") == "YEAR"],
}

with open(GRAPH_JSON, "w", encoding="utf-8") as f:
    json.dump(graph_metadata, f, indent=2, ensure_ascii=False)
print(f"  + {GRAPH_JSON}")


# =============================================================================
# GRAPH SUMMARY
# =============================================================================

print("\nGraph summary")
print(f"  Total nodes:  {G.number_of_nodes():,}")
print(f"  Total edges:  {G.number_of_edges():,}")

node_types = defaultdict(int)
for node in G.nodes:
    node_types[G.nodes[node].get("node_type", "UNKNOWN")] += 1
print(f"\n  Node breakdown:")
for ntype, count in sorted(node_types.items()):
    print(f"    {ntype}: {count}")

edge_types = defaultdict(int)
for u, v in G.edges:
    edge_types[G[u][v].get("edge_type", "UNKNOWN")] += 1
print(f"\n  Edge breakdown:")
for etype, count in sorted(edge_types.items()):
    print(f"    {etype}: {count}")

print(f"\n  Top 10 most connected speakers:")
speaker_degrees = {
    node: G.degree(node)
    for node in G.nodes
    if G.nodes[node].get("node_type") == "SPEAKER"
}
for speaker, degree in sorted(speaker_degrees.items(), key=lambda x: -x[1])[:10]:
    party_neighbours = [
        nb for nb in G.neighbors(speaker)
        if G.nodes[nb].get("node_type") == "PARTY"
    ]
    party_label = party_neighbours[0] if party_neighbours else "Unknown"
    print(f"    {speaker} ({party_label}): {degree} connections")