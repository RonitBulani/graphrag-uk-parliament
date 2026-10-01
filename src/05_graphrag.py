# GraphRAG queries: graph-guided retrieval for multi-hop Brexit questions
# Walk the knowledge graph first to find relevant MPs, then search their speeches
# Pipeline: extract entities -> score speakers -> 6-stage retrieval -> cross-encoder rerank
# -> iterative retrieval (draft, re-extract, re-retrieve) -> party diversity -> generate

import json
import csv
import time
import re
import requests
from pathlib import Path
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from openai import OpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================

_ROOT            = Path(__file__).resolve().parent.parent
GRAPH_FILE       = _ROOT / "data" / "knowledge_graph.graphml"
GRAPH_JSON       = _ROOT / "data" / "knowledge_graph_metadata.json"
VECTORSTORE_DIR  = _ROOT / "data" / "vectorstore"
QUESTIONS_FILE   = _ROOT / "data" / "questions.json"
RESULTS_FILE     = _ROOT / "results" / "graphrag_results.json"
RESULTS_CSV      = _ROOT / "results" / "graphrag_results.csv"

EMBEDDING_MODEL    = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-12-v2"

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

TOP_K_GRAPH        = 15
TOP_K_VECTOR       = 10
TOP_K_FINAL        = 5
MAX_GRAPH_SPEAKERS = 20
ITERATIVE          = True
RERANK_THRESHOLD   = -1.0  # overridden per-question by _hop_threshold()
GRAPH_BOOST_ALPHA  = 0.08  # additive graph-rank bonus on cross-encoder score

# phrases whose exact text should trigger a keyword document search (Stage 6)
_KW_TEXT_TERMS = {
    "chequers":              ["Chequers"],
    "article 50":            ["Article 50"],
    "backstop":              ["backstop"],
    "northern ireland":      ["Northern Ireland"],
    "customs union":         ["customs union"],
    "freedom of movement":   ["freedom of movement"],
    "withdrawal agreement":  ["withdrawal agreement"],
    "lancaster house":       ["Lancaster House"],
    "no-deal":               ["no-deal", "no deal"],
    "single market":         ["single market"],
    "erg":                   ["European Research Group"],
    "european research group": ["European Research Group"],
    "meaningful vote":       ["meaningful vote"],
}


# stricter threshold for simple questions, lenient for complex multi-hop ones
def _hop_threshold(question):
    q = question.lower()
    if any(w in q for w in ["what is", "how was", "define", "what are", "who is"]):
        return -0.5   # 1-hop definitional: need precise chunks
    if any(w in q for w in ["differ", "compare", "versus", " vs ", "contrast", "relate to"]):
        return -0.8   # 2-hop comparison: moderate filtering
    return -1.5       # 3-hop analytical: structural coverage matters more than precision


# loading questions from shared JSON so all scripts use identical question sets
with open(QUESTIONS_FILE, encoding="utf-8") as _qf:
    _q_data = json.load(_qf)
TEST_QUESTIONS = [q["question"] for q in _q_data]
QUESTION_IDS   = [q["id"]       for q in _q_data]


# =============================================================================
# LOADING GRAPH AND MODELS
# =============================================================================

print("Loading graph, models and pre-computing indexes...")
print("-" * 60)

# loading the knowledge graph built in step 3
G = nx.read_graphml(GRAPH_FILE)
print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

with open(GRAPH_JSON, encoding="utf-8") as f:
    meta = json.load(f)

ALL_SPEAKERS = list(meta["speakers"].keys())
ALL_PARTIES  = meta["parties"]
ALL_TOPICS   = meta["topics"]
ALL_YEARS    = meta["years"]
print(f"  Metadata: {len(ALL_SPEAKERS)} speakers, {len(ALL_PARTIES)} parties, "
      f"{len(ALL_TOPICS)} topics, {len(ALL_YEARS)} years")

model    = SentenceTransformer(EMBEDDING_MODEL)
print(f"  Bi-encoder loaded: {EMBEDDING_MODEL}")

reranker = CrossEncoder(RERANKER_MODEL)
print(f"  Cross-encoder loaded: {RERANKER_MODEL}")

client     = chromadb.PersistentClient(path=VECTORSTORE_DIR)
collection = client.get_collection("brexit_speeches")
print(f"  Vector store: {collection.count():,} chunks")

# pre-computing topic embeddings once so entity extraction is fast per question
print("  Pre-computing topic embeddings...", end=" ", flush=True)
TOPIC_EMBEDDINGS = model.encode(ALL_TOPICS, show_progress_bar=False)
print("done")

# pre-computing which years each speaker was active in (used for temporal scoring)
print("  Pre-computing speaker activity years...", end=" ", flush=True)
SPEAKER_YEARS = {}
for node in G.nodes:
    if G.nodes[node].get("node_type") != "SPEAKER":
        continue
    years = set()
    for topic_nb in G.neighbors(node):
        if G.nodes[topic_nb].get("node_type") == "TOPIC":
            for year_nb in G.neighbors(topic_nb):
                if G.nodes[year_nb].get("node_type") == "YEAR":
                    years.add(year_nb)
    SPEAKER_YEARS[node] = years
print("done")

# detecting debate communities for bridge speaker identification
print("  Detecting debate communities...", end=" ", flush=True)
debate_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "DEBATED_WITH"]
debate_subgraph = nx.Graph(debate_edges)
if debate_subgraph.number_of_nodes() > 0:
    communities  = list(nx_comm.greedy_modularity_communities(debate_subgraph))
    COMMUNITY_MAP = {node: i for i, comm in enumerate(communities) for node in comm}
else:
    communities   = []
    COMMUNITY_MAP = {}
print(f"done ({len(communities)} communities)")


# =============================================================================
# CHECKING LLM CONNECTION
# =============================================================================

print(f"\nChecking LLM connection [{PROVIDER}]...")
print("-" * 60)

llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "none")

try:
    resp   = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    print(f"  Ollama running. Models: {models}")
    if not any(OLLAMA_MODEL in m for m in models):
        print(f"  WARNING: {OLLAMA_MODEL} not found! Run: ollama pull {OLLAMA_MODEL}")
except requests.ConnectionError:
    print("  ERROR: Cannot connect to Ollama. Run: ollama serve")
    exit(1)


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

# extract graph-relevant entities (speakers, parties, years, topics) from text
def extract_entities(text, top_k_topics=15, similarity_threshold=0.25):
    q = text.lower()

    matched_speakers = [s for s in ALL_SPEAKERS if s.lower() in q]

    party_aliases = {
        "labour": "Lab", "conservative": "Con", "tory": "Con", "tories": "Con",
        "snp": "SNP", "lib dem": "LibDem", "liberal democrat": "LibDem",
        "dup": "DUP", "ukip": "UKIP", "plaid": "PlaidCymru", "green": "GPEW",
        "backbench": "Con",
    }

    # topic phrases that reliably imply specific parties even without naming them
    _KEYWORD_PARTIES = {
        "chequers":            ["Con"],
        "lancaster house":     ["Con"],
        "no-deal":             ["Con"],
        "no deal":             ["Con"],
        "customs union":       ["Lab", "Con"],
        "freedom of movement": ["Lab", "Con", "SNP"],
        "single market":       ["Lab", "Con"],
        "withdrawal agreement": ["Con"],
        "backstop":            ["DUP", "Con"],
        "northern ireland":    ["DUP", "Con"],
    }
    _KEYWORD_SPEAKERS = {
        "chequers":              ["Theresa May", "Boris Johnson", "David Davis"],
        "backstop":              ["Nigel Dodds", "Theresa May"],
        "northern ireland":      ["Nigel Dodds"],
        "customs union":         ["Keir Starmer"],
        "article 50":            ["Theresa May"],
        "erg":                   ["Jacob Rees-Mogg", "Bill Cash", "Bernard Jenkin", "Mark Francois"],
        "european research group": ["Jacob Rees-Mogg", "Bill Cash", "Bernard Jenkin", "Mark Francois"],
    }

    matched_parties = []
    for alias, code in party_aliases.items():
        if alias in q and code and code not in matched_parties:
            matched_parties.append(code)
    for p in ALL_PARTIES:
        if p.lower() in q and p not in matched_parties:
            matched_parties.append(p)
    for keyword, parties in _KEYWORD_PARTIES.items():
        if keyword in q:
            for p in parties:
                if p not in matched_parties:
                    matched_parties.append(p)
    for keyword, speakers in _KEYWORD_SPEAKERS.items():
        if keyword in q:
            for s in speakers:
                if s not in matched_speakers:
                    matched_speakers.append(s)

    matched_years = [y for y in ALL_YEARS if y in text]
    # expanding year ranges like "2017 to 2019" into individual years
    range_match   = re.search(r'\b(20\d\d)\b.*?\bto\b.*?\b(20\d\d)\b', text, re.IGNORECASE)
    if range_match:
        try:
            y_start, y_end = int(range_match.group(1)), int(range_match.group(2))
            for y in range(y_start, y_end + 1):
                if str(y) not in matched_years and str(y) in ALL_YEARS:
                    matched_years.append(str(y))
        except ValueError:
            pass

    # matching topics by cosine similarity to pre-computed embeddings
    q_embedding = model.encode([text])
    similarities = cosine_similarity(q_embedding, TOPIC_EMBEDDINGS)[0]
    top_indices  = np.argsort(similarities)[::-1]
    matched_topics = [
        ALL_TOPICS[i] for i in top_indices[:top_k_topics]
        if similarities[i] >= similarity_threshold
    ]

    return {
        "speakers":   matched_speakers,
        "parties":    matched_parties,
        "years":      matched_years,
        "topics":     matched_topics,
        "topic_sims": {
            ALL_TOPICS[i]: float(similarities[i])
            for i in top_indices[:top_k_topics]
            if similarities[i] >= similarity_threshold
        },
    }


# merge two entity dicts without duplicates (used for iterative retrieval)
def merge_entities(e1, e2):
    merged_topics = list(e1["topics"])
    for t in e2["topics"]:
        if t not in merged_topics:
            merged_topics.append(t)
    merged_sims = dict(e1["topic_sims"])
    for t, s in e2["topic_sims"].items():
        if t not in merged_sims or s > merged_sims[t]:
            merged_sims[t] = s
    return {
        "speakers":   list(dict.fromkeys(e1["speakers"] + e2["speakers"])),
        "parties":    list(dict.fromkeys(e1["parties"]  + e2["parties"])),
        "years":      list(dict.fromkeys(e1["years"]    + e2["years"])),
        "topics":     merged_topics,
        "topic_sims": merged_sims,
    }


# =============================================================================
# GRAPH TRAVERSAL
# =============================================================================

# z-score per-speaker sentiment for a topic -- sharper signal than raw polarity
def _topic_sentiment_zscores(topic):
    if not G.has_node(topic):
        return {}
    sentiments = {}
    for nb in G.neighbors(topic):
        if G.nodes[nb].get("node_type") == "SPEAKER":
            s = G[topic][nb].get("avg_sentiment", None)
            if s is not None:
                sentiments[nb] = float(s)
    if len(sentiments) < 3:
        return {}
    vals = list(sentiments.values())
    mean = sum(vals) / len(vals)
    std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    if std == 0:
        return {}
    return {spk: (s - mean) / std for spk, s in sentiments.items()}


# MPs who debated across community boundaries -- useful for multi-party questions
def get_bridge_speakers(target_parties, top_n=5):
    if len(target_parties) < 2 or not COMMUNITY_MAP:
        return []
    bridges = {}
    for speaker in ALL_SPEAKERS:
        if speaker not in COMMUNITY_MAP:
            continue
        party = next(
            (nb for nb in G.neighbors(speaker) if G.nodes[nb].get("node_type") == "PARTY"),
            None
        )
        if party not in target_parties:
            continue
        neighbour_comms = set(
            COMMUNITY_MAP[nb]
            for nb in G.neighbors(speaker)
            if G.nodes[nb].get("node_type") == "SPEAKER" and nb in COMMUNITY_MAP
        )
        if len(neighbour_comms) >= 2:
            bridges[speaker] = len(neighbour_comms)
    return sorted(bridges, key=bridges.get, reverse=True)[:top_n]


# score speakers via weighted graph traversal -- returns (ranked_names, paths_dict)
# weights: named (3.0) > party member (2.0) > topic co-speech (1.5 x sim x z x temporal)
def traverse_graph(entities, question=""):
    speaker_scores = {}
    speaker_paths  = {}
    _PROCEDURAL    = {"CHAIR", "THE CHAIR", "DEPUTY SPEAKER", "SPEAKER",
                      "THE SPEAKER", "MADAM DEPUTY SPEAKER", "MR DEPUTY SPEAKER"}

    def add_speaker(name, weight=1.0, path=""):
        if name in _PROCEDURAL:
            return
        if G.has_node(name) and G.nodes[name].get("node_type") == "SPEAKER":
            sc = G.nodes[name].get("speech_count", 1)
            speaker_scores[name] = speaker_scores.get(name, 0) + weight * (1 + sc / 500)
            if name not in speaker_paths or weight > speaker_scores.get(name, 0) * 0.5:
                speaker_paths[name] = path

    question_years = set(re.findall(r'\b(201[6-9]|202[0-1])\b', question))

    # highest-weight: speakers named explicitly in the question
    for speaker in entities["speakers"]:
        add_speaker(speaker, weight=3.0, path="Named directly in question")
        for nb in G.neighbors(speaker):
            if G.nodes[nb].get("node_type") == "SPEAKER":
                debate_count = G[speaker][nb].get("debate_count", 0)
                add_speaker(nb, weight=1.0 + debate_count / 20,
                            path=f"Debate partner of {speaker} ({debate_count} debates)")

    # party members of parties mentioned in the question
    for party in entities["parties"]:
        if not G.has_node(party):
            continue
        for nb in G.neighbors(party):
            if G.nodes[nb].get("node_type") == "SPEAKER":
                add_speaker(nb, weight=2.0, path=f"{party} member")

    # topic-matched speakers with sentiment z-score and temporal boost
    topic_sims = entities.get("topic_sims", {})
    for topic in entities["topics"]:
        if not G.has_node(topic):
            continue
        sim_weight = 1.0 + topic_sims.get(topic, 0.35)
        zscores    = _topic_sentiment_zscores(topic)

        for nb in G.neighbors(topic):
            if G.nodes[nb].get("node_type") != "SPEAKER":
                continue
            edge_count    = G[topic][nb].get("speech_count", 0)
            base_weight   = (1.5 + edge_count / 10) * sim_weight
            z             = abs(zscores.get(nb, 0))
            raw_z         = zscores.get(nb, 0)
            sentiment_boost = 1.0 + min(z * 0.3, 0.6)

            # per-topic temporal boost from year_counts on the edge
            temporal_boost = 1.0
            if question_years:
                edge_data = G[nb][topic] if G.has_edge(nb, topic) else {}
                raw_yc    = edge_data.get("year_counts", "{}")
                try:
                    topic_year_counts = json.loads(raw_yc) if isinstance(raw_yc, str) else raw_yc
                except (ValueError, TypeError):
                    topic_year_counts = {}
                topic_speeches_in_qyears = sum(topic_year_counts.get(y, 0) for y in question_years)
                if topic_speeches_in_qyears > 0:
                    temporal_boost = 1.0 + min(topic_speeches_in_qyears / 10, 0.8)

            final_weight = base_weight * sentiment_boost * temporal_boost
            stance       = "supportive" if raw_z > 0.5 else "critical" if raw_z < -0.5 else "neutral"
            speaker_active = SPEAKER_YEARS.get(nb, set())
            overlap        = len(question_years & speaker_active)
            years_note     = (f", active {overlap}/{len(question_years)} query years"
                              if question_years else "")
            add_speaker(nb, weight=final_weight,
                        path=f"Spoke about '{topic}' ({edge_count} speeches, {stance}){years_note}")

    # year-path: active speakers in matched topic × question year combinations
    matched_topic_set = set(entities.get("topics", []))
    for year in entities["years"]:
        if not G.has_node(year):
            continue
        for topic_node in G.neighbors(year):
            if G.nodes[topic_node].get("node_type") != "TOPIC":
                continue
            if topic_node not in matched_topic_set:
                continue  # only traverse year→topic paths for question-matched topics
            for nb in G.neighbors(topic_node):
                if G.nodes[nb].get("node_type") == "SPEAKER":
                    add_speaker(nb, weight=1.5, path=f"Active in {year} -> {topic_node}")

    # adding community bridge speakers for multi-party questions
    bridge_speakers = get_bridge_speakers(entities.get("parties", []))
    for bs in bridge_speakers:
        add_speaker(bs, weight=1.5, path="Cross-community bridge speaker")

    # fallback: if a requested party is unrepresented, pull its top speakers directly
    if entities.get("parties"):
        top20 = {name for name, _ in sorted(speaker_scores.items(), key=lambda x: -x[1])[:20]}
        represented_parties = set()
        for spk in top20:
            for nb in G.neighbors(spk):
                if G.nodes[nb].get("node_type") == "PARTY":
                    represented_parties.add(nb)

        missing_parties = set(entities["parties"]) - represented_parties
        fallback_needed = len(speaker_scores) < 5 or bool(missing_parties)

        if fallback_needed:
            parties_to_fill = missing_parties if missing_parties else set(entities["parties"])
            for party in parties_to_fill:
                if not G.has_node(party):
                    continue
                members = [
                    (nb, G.nodes[nb].get("speech_count", 0))
                    for nb in G.neighbors(party)
                    if G.nodes[nb].get("node_type") == "SPEAKER" and nb not in _PROCEDURAL
                ]
                for nb, sc in sorted(members, key=lambda x: -x[1])[:8]:
                    add_speaker(nb, weight=1.8, path=f"Fallback: {party} member (sc={sc})")

    # Personalized PageRank: propagate relevance through graph structure.
    # Seeded on matched entities (topics > parties > speakers > years); augments manual scores.
    if speaker_scores:
        personalization = {}
        topic_sims_local = entities.get("topic_sims", {})
        for topic in entities["topics"]:
            if G.has_node(topic):
                personalization[topic] = 1.0 + topic_sims_local.get(topic, 0.35)
        for party in entities["parties"]:
            if G.has_node(party):
                personalization[party] = 2.0
        for spk in entities["speakers"]:
            if G.has_node(spk):
                personalization[spk] = 3.0
        for yr in entities["years"]:
            if G.has_node(yr):
                personalization[yr] = 1.5
        if personalization:
            total_w = sum(personalization.values())
            personalization = {k: v / total_w for k, v in personalization.items()}
            try:
                ppr = nx.pagerank(G, alpha=0.85, personalization=personalization, max_iter=200)
                PPR_SCALE = 8.0  # additive scale: PPR values are tiny (~1e-4), boost to match traversal
                for spk in list(speaker_scores.keys()):
                    if spk in ppr:
                        speaker_scores[spk] += PPR_SCALE * ppr[spk]
            except nx.PowerIterationFailedConvergence:
                pass

    ranked       = sorted(speaker_scores.items(), key=lambda x: -x[1])
    ranked_names = [name for name, _ in ranked[:MAX_GRAPH_SPEAKERS]]
    return ranked_names, speaker_paths


# build party + stance profiles for each speaker (used in the LLM prompt)
def build_speaker_profiles(graph_speakers, entities):
    profiles   = {}
    topic_sims = entities.get("topic_sims", {})
    relevant_topics = sorted(topic_sims, key=lambda t: -topic_sims[t])[:5]

    for speaker in graph_speakers:
        if not G.has_node(speaker):
            continue
        party = next(
            (nb for nb in G.neighbors(speaker) if G.nodes[nb].get("node_type") == "PARTY"),
            "Unknown"
        )
        total_speeches = G.nodes[speaker].get("speech_count", 0)
        topic_stances  = []

        for topic in relevant_topics:
            if not G.has_edge(speaker, topic):
                continue
            edge = G[speaker][topic]
            cnt  = edge.get("speech_count", 0)
            if cnt == 0:
                continue
            zscores = _topic_sentiment_zscores(topic)
            z       = zscores.get(speaker, 0)
            stance  = "supportive" if z > 0.5 else "critical" if z < -0.5 else "neutral"
            topic_stances.append(f"{topic} ({cnt} speeches, {stance})")

        if topic_stances:
            profiles[speaker] = {
                "party": party, "total_speeches": total_speeches, "stances": topic_stances,
            }
    return profiles


# =============================================================================
# POST-PROCESSING HELPERS
# =============================================================================

# mitigate "lost in the middle" -- put highest-ranked chunks at start and end
# pattern for 5 chunks: [rank1, rank3, rank5, rank4, rank2]
def _reorder_lost_in_middle(chunks):
    if len(chunks) <= 2:
        return chunks
    result = [None] * len(chunks)
    left, right = 0, len(chunks) - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result[left] = chunk; left += 1
        else:
            result[right] = chunk; right -= 1
    return result


# speaker dedup + year diversity as a final pass over the selected top-k chunks
def _post_process(chunks, question, pool, top_k):
    # replace same-speaker duplicates with the next unique speaker from the reranked pool
    seen = set()
    diverse = []
    for c in chunks:
        if c["speaker"] not in seen:
            diverse.append(c); seen.add(c["speaker"])
    for c in pool:
        if len(diverse) >= top_k: break
        if c["speaker"] not in seen:
            diverse.append(c); seen.add(c["speaker"])

    # year diversity: guarantee one chunk per year for temporal range questions
    q_years = set(re.findall(r'\b(201[6-9]|202[0-1])\b', question))
    if len(q_years) >= 2:
        covered = {c["date"][:4] for c in diverse}
        for yr in sorted(q_years - covered):
            for c in pool:
                if c["date"][:4] == yr and c not in diverse:
                    if len(diverse) == top_k:
                        diverse[-1] = c  # replace lowest-ranked
                    else:
                        diverse.append(c)
                    diverse.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
                    break

    return diverse[:top_k]


# =============================================================================
# RETRIEVAL FUNCTION
# =============================================================================

# 6-stage retrieval then rerank with a graph-score boost
# stages: graph-filtered / global / party-targeted / pinned-speaker / year-stratified / keyword-text
# extra_embedding: optional (1 x dim) array blended with the question embedding (HyDE)
def retrieve_graphrag(question, entities, graph_speakers,
                      top_k_graph=TOP_K_GRAPH, top_k_vector=TOP_K_VECTOR,
                      exclude_texts=None, extra_embedding=None):
    q_emb_raw = model.encode([question])
    if extra_embedding is not None:
        # HyDE blend: average question embedding with hypothetical document embedding
        q_emb_raw = (q_emb_raw + extra_embedding) / 2.0
    query_embedding = q_emb_raw.tolist()
    exclude_texts   = exclude_texts or set()
    retrieved       = []

    # stage 1: graph-filtered -- only from speakers identified by graph traversal
    if graph_speakers:
        try:
            graph_results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k_graph, collection.count()),
                where={"speaker": {"$in": graph_speakers}},
            )
            for i in range(len(graph_results["documents"][0])):
                retrieved.append({
                    "text":     graph_results["documents"][0][i],
                    "speaker":  graph_results["metadatas"][0][i]["speaker"],
                    "party":    graph_results["metadatas"][0][i]["party"],
                    "date":     graph_results["metadatas"][0][i]["date"],
                    "agenda":   graph_results["metadatas"][0][i]["agenda"],
                    "distance": graph_results["distances"][0][i],
                    "source":   "graph",
                })
        except Exception as e:
            print(f"    [graph search error: {e}]")

    # stage 2: global vector search (fallback when graph misses relevant chunks)
    plain_results = collection.query(query_embeddings=query_embedding, n_results=top_k_vector)
    for i in range(len(plain_results["documents"][0])):
        retrieved.append({
            "text":     plain_results["documents"][0][i],
            "speaker":  plain_results["metadatas"][0][i]["speaker"],
            "party":    plain_results["metadatas"][0][i]["party"],
            "date":     plain_results["metadatas"][0][i]["date"],
            "agenda":   plain_results["metadatas"][0][i]["agenda"],
            "distance": plain_results["distances"][0][i],
            "source":   "vector",
        })

    # stage 3: party-targeted -- guarantees named-party chunks
    for party in entities.get("parties", []):
        try:
            party_results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(15, collection.count()),
                where={"party": {"$in": [party]}},
            )
            for i in range(len(party_results["documents"][0])):
                retrieved.append({
                    "text":     party_results["documents"][0][i],
                    "speaker":  party_results["metadatas"][0][i]["speaker"],
                    "party":    party_results["metadatas"][0][i]["party"],
                    "date":     party_results["metadatas"][0][i]["date"],
                    "agenda":   party_results["metadatas"][0][i]["agenda"],
                    "distance": party_results["distances"][0][i],
                    "source":   "party",
                })
        except Exception as e:
            print(f"    [party search error ({party}): {e}]")

    # stage 4: named-speaker pinned -- always include chunks from explicitly named speakers
    for named_spk in entities.get("speakers", []):
        try:
            pinned_results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(15, collection.count()),
                where={"speaker": {"$in": [named_spk]}},
            )
            for i in range(len(pinned_results["documents"][0])):
                retrieved.append({
                    "text":     pinned_results["documents"][0][i],
                    "speaker":  pinned_results["metadatas"][0][i]["speaker"],
                    "party":    pinned_results["metadatas"][0][i]["party"],
                    "date":     pinned_results["metadatas"][0][i]["date"],
                    "agenda":   pinned_results["metadatas"][0][i]["agenda"],
                    "distance": pinned_results["distances"][0][i],
                    "source":   "pinned",
                })
        except Exception as e:
            print(f"    [pinned search error ({named_spk}): {e}]")

    # stage 5: year-stratified -- embed year-anchored sub-queries for temporal range questions
    year_list = sorted(entities.get("years", []))
    if len(year_list) >= 2:
        for yr in year_list:
            try:
                yr_embedding = model.encode([f"{question} {yr}"]).tolist()
                yr_results   = collection.query(query_embeddings=yr_embedding, n_results=5)
                for i in range(len(yr_results["documents"][0])):
                    retrieved.append({
                        "text":     yr_results["documents"][0][i],
                        "speaker":  yr_results["metadatas"][0][i]["speaker"],
                        "party":    yr_results["metadatas"][0][i]["party"],
                        "date":     yr_results["metadatas"][0][i]["date"],
                        "agenda":   yr_results["metadatas"][0][i]["agenda"],
                        "distance": yr_results["distances"][0][i],
                        "source":   f"year_{yr}",
                    })
            except Exception as e:
                print(f"    [year search error ({yr}): {e}]")

    # stage 6: keyword text search -- exact phrase matching via where_document.
    # named speakers are already handled by Stage 4 pinned search (metadata filter);
    # adding them here returns chunks ABOUT the speaker, not BY them, hurting hit rate.
    q_lower  = question.lower()
    kw_terms = []
    for phrase, terms in _KW_TEXT_TERMS.items():
        if phrase in q_lower:
            kw_terms.extend(terms)
    for term in set(kw_terms):
        try:
            kw_results = collection.query(
                query_embeddings=query_embedding,
                n_results=5,
                where_document={"$contains": term},
            )
            for i in range(len(kw_results["documents"][0])):
                retrieved.append({
                    "text":     kw_results["documents"][0][i],
                    "speaker":  kw_results["metadatas"][0][i]["speaker"],
                    "party":    kw_results["metadatas"][0][i]["party"],
                    "date":     kw_results["metadatas"][0][i]["date"],
                    "agenda":   kw_results["metadatas"][0][i]["agenda"],
                    "distance": kw_results["distances"][0][i],
                    "source":   f"kw_{term[:12]}",
                })
        except Exception:
            pass

    # deduplicating by text prefix and (speaker, date) pair
    seen_texts        = {}
    seen_speaker_dates = {}
    for chunk in retrieved:
        txt          = chunk["text"].strip()
        text_key     = txt[:80]
        spk_date_key = (chunk["speaker"], chunk["date"])
        if text_key in exclude_texts:
            continue
        if spk_date_key in seen_speaker_dates:
            existing = seen_speaker_dates[spk_date_key]
            if chunk["distance"] >= seen_texts[existing]["distance"]:
                continue
            del seen_texts[existing]
        if text_key not in seen_texts or chunk["distance"] < seen_texts[text_key]["distance"]:
            seen_texts[text_key]          = chunk
            seen_speaker_dates[spk_date_key] = text_key

    deduped = list(seen_texts.values())

    threshold = _hop_threshold(question)
    if deduped:
        pairs = [[question, c["text"]] for c in deduped]
        scores = reranker.predict(pairs)

        # graph-score boost: speakers ranked higher by graph traversal get a small bonus
        graph_rank = {spk: i for i, spk in enumerate(graph_speakers)}
        n_graph    = max(len(graph_speakers), 1)
        boosted    = []
        for raw_score, chunk in zip(scores, deduped):
            spk = chunk["speaker"]
            if spk in graph_rank:
                rank_norm    = 1.0 - graph_rank[spk] / n_graph
                final_score  = float(raw_score) + GRAPH_BOOST_ALPHA * rank_norm
            else:
                final_score  = float(raw_score)
            boosted.append((final_score, chunk))

        ranked = sorted(boosted, key=lambda x: -x[0])
        all_ranked_chunks = [dict(c, rerank_score=s) for s, c in ranked]
        filtered = [dict(c, rerank_score=s) for s, c in ranked if s >= threshold]
        deduped = filtered if filtered else all_ranked_chunks[:TOP_K_FINAL]

        # named speaker guarantee: speakers explicitly named always get at least 1 chunk
        named_spks = entities.get("speakers", [])
        if named_spks:
            present = {c["speaker"] for c in deduped}
            for spk in named_spks:
                if spk not in present:
                    best = next((c for c in all_ranked_chunks if c["speaker"] == spk), None)
                    if best:
                        deduped.append(best)
                        present.add(spk)

    # applying party diversity enforcement
    target_parties = entities.get("parties", [])
    if len(target_parties) >= 2:
        per_party_cap = max(1, -(-TOP_K_FINAL // len(target_parties)))
        party_counts  = {p: 0 for p in target_parties}
        final         = []
        remaining     = list(deduped)

        for party in target_parties:
            for chunk in remaining:
                if chunk["party"] == party:
                    final.append(chunk)
                    party_counts[party] = party_counts.get(party, 0) + 1
                    remaining.remove(chunk)
                    break

        for chunk in remaining:
            if len(final) >= TOP_K_FINAL: break
            p = chunk["party"]
            if p in party_counts and party_counts[p] >= per_party_cap: continue
            final.append(chunk)
            party_counts[p] = party_counts.get(p, 0) + 1

        for chunk in remaining:
            if len(final) >= TOP_K_FINAL: break
            if chunk not in final:
                final.append(chunk)

        final.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
        return _post_process(final[:TOP_K_FINAL], question, deduped, TOP_K_FINAL)

    elif len(target_parties) == 1:
        target_party = target_parties[0]
        min_target   = max(1, -(-TOP_K_FINAL // 2))
        final        = []
        remaining    = list(deduped)

        for chunk in remaining[:]:
            if len(final) >= min_target: break
            if chunk["party"] == target_party:
                final.append(chunk); remaining.remove(chunk)

        for chunk in remaining:
            if len(final) >= TOP_K_FINAL: break
            final.append(chunk)

        final.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
        return _post_process(final[:TOP_K_FINAL], question, deduped, TOP_K_FINAL)

    return _post_process(deduped[:TOP_K_FINAL], question, deduped, TOP_K_FINAL)


# =============================================================================
# ANSWER GENERATION
# =============================================================================

# quick 150-token draft for iterative entity extraction
def generate_interim(question, context_chunks):
    context = ""
    for i, chunk in enumerate(context_chunks[:3]):
        context += (f"\n--- Speech {i+1} ({chunk['speaker']}, {chunk['party']}, "
                    f"{chunk['date']}) ---\n")
        context += chunk["text"] + "\n"

    for attempt in range(3):
        try:
            resp = llm_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content":
                    f"Based on these parliamentary speeches, briefly answer:\n\n{context}\n\n"
                    f"QUESTION: {question}\n\n"
                    f"BRIEF ANSWER (2-3 sentences, name specific MPs, parties, topics, years):"}],
                temperature=0.1,
                max_tokens=150,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt < 2:
                time.sleep(5)
            else:
                return ""


# build a prompt using speaker profiles and graph paths, then generate a final answer
def generate(question, context_chunks, speaker_profiles, speaker_paths):
    # apply lost-in-middle reordering before building the context string
    context_chunks = _reorder_lost_in_middle(context_chunks)
    context = ""
    for i, chunk in enumerate(context_chunks):
        spk  = chunk["speaker"]
        path = speaker_paths.get(spk, "plain vector search")
        context += (f"\n--- Speech {i+1} ({spk}, {chunk['party']}, {chunk['date']}) ---\n"
                    f"[Retrieved via: {path}]\n")
        context += chunk["text"] + "\n"

    # including graph-derived stance summaries in the prompt
    profiles_text = ""
    for speaker in {c["speaker"] for c in context_chunks}:
        if speaker in speaker_profiles:
            p      = speaker_profiles[speaker]
            stances = "; ".join(p["stances"]) if p["stances"] else "general Brexit debates"
            profiles_text += (
                f"  {speaker} ({p['party']}, {p['total_speeches']} total speeches): {stances}\n"
            )

    profiles_section = (
        f"\nSPEAKER PROFILES (from parliamentary record):\n{profiles_text}"
        if profiles_text else ""
    )

    # tailoring task instruction by question type
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
        task_instruction = "Answer comprehensively, citing specific speakers, dates, and stances."

    prompt = f"""You are a research assistant analysing UK parliamentary speeches about Brexit.
{profiles_section}
Answer the following question based ONLY on the parliamentary speeches provided below.
Use specific details, speaker names, dates, and their known stances from the profiles above.
Where speakers have different tones (supportive vs critical), highlight those differences.
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
                max_tokens=768,
            )
            return response.choices[0].message.content or "ERROR: No response from LLM"
        except Exception as e:
            if attempt < 2:
                print(f"  [generate retry {attempt+1}: {type(e).__name__}]")
                time.sleep(5)
            else:
                return "ERROR: LLM unavailable after retries"


# =============================================================================
# RUNNING QUESTIONS
# =============================================================================

# list of question IDs for patch mode (re-run a subset); empty list runs all
PATCH_QUESTION_IDS = []

print(f"\nRunning {len(TEST_QUESTIONS)} questions through GraphRAG...")
print("=" * 60)

# in patch mode, loading existing results and only re-running specified questions
_existing_results = []
if PATCH_QUESTION_IDS and RESULTS_FILE.exists():
    with open(RESULTS_FILE, encoding="utf-8") as _pf:
        _existing_results = json.load(_pf)
    print(f"  Patch mode: re-running Q{PATCH_QUESTION_IDS} only (loaded {len(_existing_results)} existing results)")
all_results = []

for i, question in enumerate(TEST_QUESTIONS):
    if PATCH_QUESTION_IDS and QUESTION_IDS[i] not in PATCH_QUESTION_IDS:
        continue
    print(f"\n  Q{i+1}/{len(TEST_QUESTIONS)}: {question}")
    start_time = time.time()

    # "What is X" questions: skipping graph traversal and using plain vector search
    _is_definitional = bool(re.match(r"^what is\b", question.strip().lower()))

    entities = extract_entities(question)
    print(f"    Entities: speakers={entities['speakers']}, parties={entities['parties']}, "
          f"years={entities['years']}, topics={len(entities['topics'])} matched")

    if _is_definitional:
        print("    [Definitional query - bypassing graph traversal]")
        graph_speakers = []
        speaker_paths  = {}
    else:
        graph_speakers, speaker_paths = traverse_graph(entities, question)
        print(f"    Traversal: {len(graph_speakers)} speakers | top-5: {graph_speakers[:5]}")

    speaker_profiles = build_speaker_profiles(graph_speakers, entities)
    chunks_r1        = retrieve_graphrag(question, entities, graph_speakers)
    r1_texts         = {c["text"].strip()[:80] for c in chunks_r1}

    final_chunks = chunks_r1
    r2_count     = 0

    # iterative retrieval: generate a draft answer, extract new entities, run a second pass
    if ITERATIVE and len(chunks_r1) > 0 and not _is_definitional:
        print("    Iterative retrieval: generating interim answer...", end=" ", flush=True)
        try:
            interim = generate_interim(question, chunks_r1)
        except Exception as _iter_err:
            print(f"skipped ({type(_iter_err).__name__})")
            interim = ""
        new_entities    = extract_entities(interim)
        merged_entities = merge_entities(entities, new_entities)

        new_topics       = [t for t in merged_entities["topics"] if t not in entities["topics"]]
        new_speakers_list = [s for s in merged_entities["speakers"] if s not in entities["speakers"]]
        print(f"expanded ({len(new_topics)} new topics, {len(new_speakers_list)} new speakers)")

        graph_speakers_r2, paths_r2 = traverse_graph(merged_entities, question)
        speaker_paths.update(paths_r2)
        speaker_profiles.update(build_speaker_profiles(graph_speakers_r2, merged_entities))

        # HyDE: encoding interim answer and blending its embedding with the question for R2
        hyde_emb = None
        if interim:
            hyde_emb = model.encode([interim])

        chunks_r2 = retrieve_graphrag(
            question, merged_entities, graph_speakers_r2,
            exclude_texts=r1_texts, extra_embedding=hyde_emb
        )

        # merging R1+R2 and re-applying diversity using original entities
        combined = list(chunks_r1) + [c for c in chunks_r2
                                       if c["text"].strip()[:80] not in r1_texts]
        combined.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

        target_parties = entities.get("parties", [])
        if target_parties:
            _per_cap = max(1, -(-TOP_K_FINAL // max(1, len(target_parties))))
            _final_c, _pc = [], {}
            for _party in target_parties:
                for _c in combined:
                    if _c["party"] == _party and _c not in _final_c:
                        _final_c.append(_c); _pc[_party] = 1; break
            for _c in combined:
                if len(_final_c) >= TOP_K_FINAL: break
                if _c in _final_c: continue
                _p = _c["party"]
                if _p in _pc and _pc[_p] >= _per_cap: continue
                _final_c.append(_c); _pc[_p] = _pc.get(_p, 0) + 1
            for _c in combined:
                if len(_final_c) >= TOP_K_FINAL: break
                if _c not in _final_c: _final_c.append(_c)
            _final_c.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
            final_chunks = _post_process(_final_c[:TOP_K_FINAL], question, combined, TOP_K_FINAL)
        else:
            final_chunks = _post_process(combined[:TOP_K_FINAL], question, combined, TOP_K_FINAL)
        r2_count = len([c for c in final_chunks if c not in chunks_r1])

    retrieve_time = time.time() - start_time
    graph_count   = sum(1 for c in final_chunks if c.get("source") == "graph")
    vector_count  = sum(1 for c in final_chunks if c.get("source") == "vector")
    print(f"    Retrieved {len(final_chunks)} chunks "
          f"({graph_count} graph, {vector_count} vector, {r2_count} from iter-2) "
          f"in {retrieve_time:.2f}s")

    for j, chunk in enumerate(final_chunks):
        print(f"    Chunk {j+1} [{chunk.get('source', '?')}]: "
              f"{chunk['speaker']} ({chunk['party']}, {chunk['date']}) "
              f"rerank={chunk.get('rerank_score', 0):.3f}")

    print("    Generating answer...")
    gen_start = time.time()
    answer    = generate(question, final_chunks, speaker_profiles, speaker_paths)
    gen_time  = time.time() - gen_start
    print(f"    Generated in {gen_time:.1f}s")
    print(f"    Answer: {answer[:200]}...")

    all_results.append({
        "question_id":           QUESTION_IDS[i],
        "question":              question,
        "answer":                answer,
        "retrieved_chunks":      final_chunks,
        "entities_matched":      {k: v for k, v in entities.items() if k != "topic_sims"},
        "topic_similarities":    entities.get("topic_sims", {}),
        "speaker_profiles":      speaker_profiles,
        "speaker_paths":         speaker_paths,
        "graph_speakers_used":   graph_speakers,
        "graph_chunks":          graph_count,
        "vector_chunks":         vector_count,
        "iterative_r2_chunks":   r2_count,
        "retrieve_time_seconds": round(retrieve_time, 3),
        "generate_time_seconds": round(gen_time, 3),
        "model":                 OLLAMA_MODEL,
        "provider":              PROVIDER,
        "top_k":                 TOP_K_FINAL,
        "system":                "graphrag",
        "enhancements": [
            "semantic_topic_matching", "zscore_sentiment", "tone_profiles",
            "party_diversity", "temporal_boost", "path_provenance",
            "community_bridges", "iterative_retrieval", "cross_encoder_rerank",
            "personalized_pagerank", "hyde_r2_embedding", "lost_in_middle_reorder",
        ],
    })


# =============================================================================
# SAVING RESULTS
# =============================================================================

print(f"\n{'=' * 60}")
print("Saving results...")
print(f"{'=' * 60}")

# merging with existing results in patch mode
if PATCH_QUESTION_IDS and _existing_results:
    patched_set = set(PATCH_QUESTION_IDS)
    merged = [r for r in _existing_results if r["question_id"] not in patched_set] + all_results
    all_results = sorted(merged, key=lambda r: r["question_id"])

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"  + {RESULTS_FILE}  ({len(all_results)} results)")

total_retrieve = sum(r["retrieve_time_seconds"] for r in all_results)
total_generate = sum(r["generate_time_seconds"] for r in all_results)
graph_guided   = sum(r["graph_chunks"] for r in all_results)
vector_only    = sum(r["vector_chunks"] for r in all_results)
iter_r2        = sum(r.get("iterative_r2_chunks", 0) for r in all_results)

print(f"  Total retrieval time:    {total_retrieve:.1f}s")
print(f"  Total generation time:   {total_generate:.1f}s")
print(f"  Avg per question:        {(total_retrieve + total_generate) / len(all_results):.1f}s")
print(f"  Graph-guided chunks:     {graph_guided}")
print(f"  Vector fallback chunks:  {vector_only}")
print(f"  Iterative R2 chunks:     {iter_r2}")

# saving a flat CSV for quick inspection
csv_cols = [
    "question_id", "question", "answer",
    "graph_chunks", "vector_chunks", "iterative_r2_chunks",
    "retrieve_time_seconds", "generate_time_seconds",
    "entities_speakers", "entities_parties", "entities_years", "entities_topics",
    "chunk1_speaker", "chunk1_party", "chunk1_date", "chunk1_source", "chunk1_rerank",
    "chunk2_speaker", "chunk2_party", "chunk2_date", "chunk2_source", "chunk2_rerank",
    "chunk3_speaker", "chunk3_party", "chunk3_date", "chunk3_source", "chunk3_rerank",
    "chunk4_speaker", "chunk4_party", "chunk4_date", "chunk4_source", "chunk4_rerank",
    "chunk5_speaker", "chunk5_party", "chunk5_date", "chunk5_source", "chunk5_rerank",
]

with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_cols)
    writer.writeheader()
    for r in all_results:
        ents = r.get("entities_matched", {})
        row  = {
            "question_id": r["question_id"], "question": r["question"],
            "answer": r["answer"],
            "graph_chunks": r.get("graph_chunks", 0),
            "vector_chunks": r.get("vector_chunks", 0),
            "iterative_r2_chunks": r.get("iterative_r2_chunks", 0),
            "retrieve_time_seconds": r["retrieve_time_seconds"],
            "generate_time_seconds": r["generate_time_seconds"],
            "entities_speakers": "; ".join(ents.get("speakers", [])),
            "entities_parties":  "; ".join(ents.get("parties", [])),
            "entities_years":    "; ".join(ents.get("years", [])),
            "entities_topics":   len(ents.get("topics", [])),
        }
        for j in range(5):
            prefix = f"chunk{j+1}_"
            if j < len(r["retrieved_chunks"]):
                c = r["retrieved_chunks"][j]
                row[prefix + "speaker"] = c.get("speaker", "")
                row[prefix + "party"]   = c.get("party", "")
                row[prefix + "date"]    = c.get("date", "")
                row[prefix + "source"]  = c.get("source", "")
                row[prefix + "rerank"]  = round(c.get("rerank_score", 0), 4)
            else:
                row[prefix + "speaker"] = row[prefix + "party"] = ""
                row[prefix + "date"]    = row[prefix + "source"] = ""
                row[prefix + "rerank"]  = ""
        writer.writerow(row)

print(f"  + {RESULTS_CSV}")