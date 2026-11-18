from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from networkx.algorithms import community as nx_comm
from openai import OpenAI
import numpy as np
import json
import networkx as nx
import re
import chromadb
import time

# Number of final evidence passages returned by retrieval.
TOP_K_FINAL = 5

# Exact phrases used by Stage 6 keyword-text retrieval.
# Preserved from the original research implementation.
_KW_TEXT_TERMS = {
    "chequers": ["Chequers"],
    "article 50": ["Article 50"],
    "backstop": ["backstop"],
    "northern ireland": ["Northern Ireland"],
    "customs union": ["customs union"],
    "freedom of movement": ["freedom of movement"],
    "withdrawal agreement": ["withdrawal agreement"],
    "lancaster house": ["Lancaster House"],
    "no-deal": ["no-deal", "no deal"],
    "single market": ["single market"],
    "erg": ["European Research Group"],
    "european research group": ["European Research Group"],
    "meaningful vote": ["meaningful vote"],
}

# Select the original relevance threshold based on question type.
def _hop_threshold(question):
    q = question.lower()

    if any(
        w in q
        for w in ["what is", "how was", "define", "what are", "who is"]
    ):
        return -0.5

    if any(
        w in q
        for w in ["differ", "compare", "versus", " vs ", "contrast", "relate to"]
    ):
        return -0.8

    return -1.5

# Final speaker and year diversity post-processing.
# Reproduced from the original GraphRAG research implementation.
def _post_process(chunks, question, pool, top_k):
    # First pass: retain one passage per speaker.
    seen = set()
    diverse = []

    for c in chunks:
        if c["speaker"] not in seen:
            diverse.append(c)
            seen.add(c["speaker"])

    # Fill remaining positions with new speakers from the pool.
    for c in pool:
        if len(diverse) >= top_k:
            break

        if c["speaker"] not in seen:
            diverse.append(c)
            seen.add(c["speaker"])

    # Attempt year diversity for temporal range questions.
    q_years = set(
        re.findall(r'\b(201[6-9]|202[0-1])\b', question)
    )

    if len(q_years) >= 2:
        covered = {
            c["date"][:4]
            for c in diverse
        }

        for yr in sorted(q_years - covered):
            for c in pool:
                if c["date"][:4] == yr and c not in diverse:
                    if len(diverse) == top_k:
                        diverse[-1] = c
                    else:
                        diverse.append(c)

                    diverse.sort(
                        key=lambda x: x.get("rerank_score", 0),
                        reverse=True,
                    )
                    break

    return diverse[:top_k]

# Merge entities from the original question and interim answer.
# Preserves the original research implementation.
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
        "speakers": list(
            dict.fromkeys(e1["speakers"] + e2["speakers"])
        ),
        "parties": list(
            dict.fromkeys(e1["parties"] + e2["parties"])
        ),
        "years": list(
            dict.fromkeys(e1["years"] + e2["years"])
        ),
        "topics": merged_topics,
        "topic_sims": merged_sims,
    }

# Reorder evidence to mitigate the "lost in the middle" effect.
# Preserves the original research implementation.
def _reorder_lost_in_middle(chunks):
    if len(chunks) <= 2:
        return chunks

    result = [None] * len(chunks)
    left, right = 0, len(chunks) - 1

    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result[left] = chunk
            left += 1
        else:
            result[right] = chunk
            right -= 1

    return result

class GraphRAGEngine:
    def __init__(self, graph_path: Path, metadata_path: Path):
        self.graph_path = graph_path
        self.metadata_path = metadata_path

        # Load the knowledge graph
        self.graph = nx.read_graphml(self.graph_path)

        # Load supporting metadata
        with self.metadata_path.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)
        
        # Prepare entity collections using the original GraphRAG logic
        self.all_speakers = list(self.metadata["speakers"].keys())
        self.all_parties = self.metadata["parties"]
        self.all_topics = self.metadata["topics"]
        self.all_years = self.metadata["years"]
        
        # Load the original embedding model once per engine instance
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        # Load the original cross-encoder reranking model.
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        
        # Connect to the original research LLM through Ollama.
        self.ollama_model = "qwen2.5:3b"

        self.llm_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=180.0,
            max_retries=0,
        )

        # Precompute topic embeddings in the same order as the metadata
        self.topic_embeddings = self.model.encode(
            self.all_topics,
            show_progress_bar=False
        )
        
        # Connect to the copied research vector store.
        self.vectorstore_path = self.graph_path.parent / "vectorstore"

        if not self.vectorstore_path.is_dir():
            raise FileNotFoundError(
                f"Vector store directory not found: {self.vectorstore_path}"
            )

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vectorstore_path)
        )

        self.collection = self.chroma_client.get_collection(
            "brexit_speeches"
        )
        
        # Precompute speaker activity years using the original graph logic
        self.speaker_years = {}

        for node in self.graph.nodes:
            if self.graph.nodes[node].get("node_type") != "SPEAKER":
                continue

            years = set()

            for topic_nb in self.graph.neighbors(node):
                if self.graph.nodes[topic_nb].get("node_type") == "TOPIC":
                    for year_nb in self.graph.neighbors(topic_nb):
                        if self.graph.nodes[year_nb].get("node_type") == "YEAR":
                            years.add(year_nb)

            self.speaker_years[node] = years
            
        # Detect communities using debate relationships
        debate_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == "DEBATED_WITH"
        ]

        debate_subgraph = nx.Graph(debate_edges)

        if debate_subgraph.number_of_nodes() > 0:
            communities = list(
                nx_comm.greedy_modularity_communities(debate_subgraph)
            )

            self.community_map = {
                node: i
                for i, comm in enumerate(communities)
                for node in comm
            }
        else:
            communities = []
            self.community_map = {}

        self.community_count = len(communities)

    def graph_info(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "graph_type": type(self.graph).__name__,
            "metadata_keys": list(self.metadata.keys()),
            "speakers": len(self.all_speakers),
            "parties": len(self.all_parties),
            "topics": len(self.all_topics),
            "years": len(self.all_years),
        }
        
    def embedding_info(self) -> dict:
        return {
            "model": "BAAI/bge-small-en-v1.5",
            "topic_count": len(self.all_topics),
            "embedding_shape": self.topic_embeddings.shape,
        }
        
    def match_topics(self, text: str, top_k_topics: int = 15, similarity_threshold: float = 0.25,
    ) -> dict:

        q_embedding = self.model.encode([text])

        similarities = cosine_similarity(
            q_embedding,
            self.topic_embeddings
        )[0]

        top_indices = np.argsort(similarities)[::-1]

        matched_topics = [
            self.all_topics[i]
            for i in top_indices[:top_k_topics]
            if similarities[i] >= similarity_threshold
        ]

        topic_sims = {
            self.all_topics[i]: float(similarities[i])
            for i in top_indices[:top_k_topics]
            if similarities[i] >= similarity_threshold
        }

        return {
            "topics": matched_topics,
            "topic_sims": topic_sims,
        }
        
    def extract_entities(self, text: str, top_k_topics: int = 15, similarity_threshold: float = 0.25,
    ) -> dict:

        q = text.lower()

        matched_speakers = [
            s for s in self.all_speakers
            if s.lower() in q
        ]

        party_aliases = {
            "labour": "Lab",
            "conservative": "Con",
            "tory": "Con",
            "tories": "Con",
            "snp": "SNP",
            "lib dem": "LibDem",
            "liberal democrat": "LibDem",
            "dup": "DUP",
            "ukip": "UKIP",
            "plaid": "PlaidCymru",
            "green": "GPEW",
            "backbench": "Con",
        }

        _KEYWORD_PARTIES = {
            "chequers": ["Con"],
            "lancaster house": ["Con"],
            "no-deal": ["Con"],
            "no deal": ["Con"],
            "customs union": ["Lab", "Con"],
            "freedom of movement": ["Lab", "Con", "SNP"],
            "single market": ["Lab", "Con"],
            "withdrawal agreement": ["Con"],
            "backstop": ["DUP", "Con"],
            "northern ireland": ["DUP", "Con"],
        }

        _KEYWORD_SPEAKERS = {
            "chequers": ["Theresa May", "Boris Johnson", "David Davis"],
            "backstop": ["Nigel Dodds", "Theresa May"],
            "northern ireland": ["Nigel Dodds"],
            "customs union": ["Keir Starmer"],
            "article 50": ["Theresa May"],
            "erg": [
                "Jacob Rees-Mogg",
                "Bill Cash",
                "Bernard Jenkin",
                "Mark Francois",
            ],
            "european research group": [
                "Jacob Rees-Mogg",
                "Bill Cash",
                "Bernard Jenkin",
                "Mark Francois",
            ],
        }

        matched_parties = []

        for alias, code in party_aliases.items():
            if alias in q and code and code not in matched_parties:
                matched_parties.append(code)

        for p in self.all_parties:
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

        matched_years = [
            y for y in self.all_years
            if y in text
        ]

        range_match = re.search(
            r'\b(20\d\d)\b.*?\bto\b.*?\b(20\d\d)\b',
            text,
            re.IGNORECASE,
        )

        if range_match:
            try:
                y_start = int(range_match.group(1))
                y_end = int(range_match.group(2))

                for y in range(y_start, y_end + 1):
                    if (
                        str(y) not in matched_years
                        and str(y) in self.all_years
                    ):
                        matched_years.append(str(y))

            except ValueError:
                pass

        topic_result = self.match_topics(
            text,
            top_k_topics=top_k_topics,
            similarity_threshold=similarity_threshold,
        )

        return {
            "speakers": matched_speakers,
            "parties": matched_parties,
            "years": matched_years,
            "topics": topic_result["topics"],
            "topic_sims": topic_result["topic_sims"],
        }    
        
    def traversal_info(self) -> dict:
        return {
            "speaker_year_mappings": len(self.speaker_years),
            "debate_community_members": len(self.community_map),
            "communities": self.community_count,
            "max_graph_speakers": 20,
        }
        
    def _topic_sentiment_zscores(self, topic):
        if not self.graph.has_node(topic):
            return {}

        sentiments = {}

        for nb in self.graph.neighbors(topic):
            if self.graph.nodes[nb].get("node_type") == "SPEAKER":
                s = self.graph[topic][nb].get("avg_sentiment", None)

                if s is not None:
                    sentiments[nb] = float(s)

        if len(sentiments) < 3:
            return {}

        vals = list(sentiments.values())

        mean = sum(vals) / len(vals)

        std = (
            sum((v - mean) ** 2 for v in vals) / len(vals)
        ) ** 0.5

        if std == 0:
            return {}

        return {
            spk: (s - mean) / std
            for spk, s in sentiments.items()
        }
        
    def get_bridge_speakers(self, target_parties, top_n=5):
        if len(target_parties) < 2 or not self.community_map:
            return []

        bridges = {}

        for speaker in self.all_speakers:
            if speaker not in self.community_map:
                continue

            party = next(
                (
                    nb
                    for nb in self.graph.neighbors(speaker)
                    if self.graph.nodes[nb].get("node_type") == "PARTY"
                ),
                None,
            )

            if party not in target_parties:
                continue

            neighbour_comms = set(
                self.community_map[nb]
                for nb in self.graph.neighbors(speaker)
                if self.graph.nodes[nb].get("node_type") == "SPEAKER"
                and nb in self.community_map
            )

            if len(neighbour_comms) >= 2:
                bridges[speaker] = len(neighbour_comms)

        return sorted(
            bridges,
            key=bridges.get,
            reverse=True,
        )[:top_n]
        
    def traverse_graph(self, entities, question=""):
        speaker_scores = {}
        speaker_paths = {}

        _PROCEDURAL = {
            "CHAIR",
            "THE CHAIR",
            "DEPUTY SPEAKER",
            "SPEAKER",
            "THE SPEAKER",
            "MADAM DEPUTY SPEAKER",
            "MR DEPUTY SPEAKER",
        }

        def add_speaker(name, weight=1.0, path=""):
            if name in _PROCEDURAL:
                return

            if (
                self.graph.has_node(name)
                and self.graph.nodes[name].get("node_type") == "SPEAKER"
            ):
                sc = self.graph.nodes[name].get("speech_count", 1)

                speaker_scores[name] = (
                    speaker_scores.get(name, 0)
                    + weight * (1 + sc / 500)
                )

                if (
                    name not in speaker_paths
                    or weight > speaker_scores.get(name, 0) * 0.5
                ):
                    speaker_paths[name] = path
                    
        # Stage 1: Score matched speakers and their debate partners
        for speaker in entities["speakers"]:
            add_speaker(
                speaker,
                weight=3.0,
                path="Named directly in question",
            )

            for nb in self.graph.neighbors(speaker):
                if self.graph.nodes[nb].get("node_type") == "SPEAKER":
                    debate_count = self.graph[speaker][nb].get(
                        "debate_count", 0
                    )

                    add_speaker(
                        nb,
                        weight=1.0 + debate_count / 20,
                        path=f"Debate partner of {speaker} ({debate_count} debates)",
                    )
                    
        # Stage 2: Score members of parties mentioned in the question
        for party in entities["parties"]:
            if not self.graph.has_node(party):
                continue

            for nb in self.graph.neighbors(party):
                if self.graph.nodes[nb].get("node_type") == "SPEAKER":
                    add_speaker(
                        nb,
                        weight=2.0,
                        path=f"{party} member",
                    )
                    
        # Stage 3: Score speakers connected to matched topics
        question_years = set(
            re.findall(r'\b(201[6-9]|202[0-1])\b', question)
        )

        topic_sims = entities.get("topic_sims", {})

        for topic in entities["topics"]:
            if not self.graph.has_node(topic):
                continue

            sim_weight = 1.0 + topic_sims.get(topic, 0.35)
            zscores = self._topic_sentiment_zscores(topic)

            for nb in self.graph.neighbors(topic):
                if self.graph.nodes[nb].get("node_type") != "SPEAKER":
                    continue

                edge_count = self.graph[topic][nb].get("speech_count", 0)
                base_weight = (1.5 + edge_count / 10) * sim_weight

                z = abs(zscores.get(nb, 0))
                raw_z = zscores.get(nb, 0)
                sentiment_boost = 1.0 + min(z * 0.3, 0.6)

                # Temporal boost based on edge-specific speech years
                temporal_boost = 1.0

                if question_years:
                    edge_data = (
                        self.graph[nb][topic]
                        if self.graph.has_edge(nb, topic)
                        else {}
                    )

                    raw_yc = edge_data.get("year_counts", "{}")

                    try:
                        topic_year_counts = (
                            json.loads(raw_yc)
                            if isinstance(raw_yc, str)
                            else raw_yc
                        )
                    except (ValueError, TypeError):
                        topic_year_counts = {}

                    topic_speeches_in_qyears = sum(
                        topic_year_counts.get(y, 0)
                        for y in question_years
                    )

                    if topic_speeches_in_qyears > 0:
                        temporal_boost = 1.0 + min(
                            topic_speeches_in_qyears / 10,
                            0.8,
                        )

                final_weight = (
                    base_weight * sentiment_boost * temporal_boost
                )

                stance = (
                    "supportive" if raw_z > 0.5
                    else "critical" if raw_z < -0.5
                    else "neutral"
                )

                speaker_active = self.speaker_years.get(nb, set())
                overlap = len(question_years & speaker_active)

                years_note = (
                    f", active {overlap}/{len(question_years)} query years"
                    if question_years else ""
                )

                add_speaker(nb, weight=final_weight, path=(f"Spoke about '{topic}' "
                        f"({edge_count} speeches, {stance}){years_note}"
                    ),
                )
        
        # Stage 4: Traverse matched year-topic-speaker paths
        matched_topic_set = set(entities.get("topics", []))

        for year in entities["years"]:
            if not self.graph.has_node(year):
                continue

            for topic_node in self.graph.neighbors(year):
                if self.graph.nodes[topic_node].get("node_type") != "TOPIC":
                    continue

                if topic_node not in matched_topic_set:
                    continue

                for nb in self.graph.neighbors(topic_node):
                    if self.graph.nodes[nb].get("node_type") == "SPEAKER":
                        add_speaker(
                            nb,
                            weight=1.5,
                            path=f"Active in {year} -> {topic_node}",
                        )
                        
        # Stage 5: Add community bridge speakers
        bridge_speakers = self.get_bridge_speakers(
            entities.get("parties", [])
        )

        for bs in bridge_speakers:
            add_speaker(
                bs,
                weight=1.5,
                path="Cross-community bridge speaker",
            )
            
        # Stage 6: Fallback for underrepresented requested parties
        if entities.get("parties"):
            top20 = {
                name
                for name, _ in sorted(
                    speaker_scores.items(),
                    key=lambda x: -x[1]
                )[:20]
            }

            represented_parties = set()

            for spk in top20:
                for nb in self.graph.neighbors(spk):
                    if self.graph.nodes[nb].get("node_type") == "PARTY":
                        represented_parties.add(nb)

            missing_parties = (
                set(entities["parties"]) - represented_parties
            )

            fallback_needed = (
                len(speaker_scores) < 5
                or bool(missing_parties)
            )

            if fallback_needed:
                parties_to_fill = (
                    missing_parties
                    if missing_parties
                    else set(entities["parties"])
                )

                for party in parties_to_fill:
                    if not self.graph.has_node(party):
                        continue

                    members = [
                        (
                            nb,
                            self.graph.nodes[nb].get("speech_count", 0)
                        )
                        for nb in self.graph.neighbors(party)
                        if (
                            self.graph.nodes[nb].get("node_type") == "SPEAKER"
                            and nb not in _PROCEDURAL
                        )
                    ]

                    for nb, sc in sorted(
                        members,
                        key=lambda x: -x[1]
                    )[:8]:
                        add_speaker(
                            nb,
                            weight=1.8,
                            path=f"Fallback: {party} member (sc={sc})",
                        )
                        
        # Stage 7: Personalized PageRank
        if speaker_scores:
            personalization = {}
            topic_sims_local = entities.get("topic_sims", {})

            for topic in entities["topics"]:
                if self.graph.has_node(topic):
                    personalization[topic] = (
                        1.0 + topic_sims_local.get(topic, 0.35)
                    )

            for party in entities["parties"]:
                if self.graph.has_node(party):
                    personalization[party] = 2.0

            for spk in entities["speakers"]:
                if self.graph.has_node(spk):
                    personalization[spk] = 3.0

            for yr in entities["years"]:
                if self.graph.has_node(yr):
                    personalization[yr] = 1.5

            if personalization:
                total_w = sum(personalization.values())

                personalization = {
                    k: v / total_w
                    for k, v in personalization.items()
                }

                try:
                    ppr = nx.pagerank(
                        self.graph,
                        alpha=0.85,
                        personalization=personalization,
                        max_iter=200,
                    )

                    PPR_SCALE = 8.0

                    for spk in list(speaker_scores.keys()):
                        if spk in ppr:
                            speaker_scores[spk] += PPR_SCALE * ppr[spk]

                except nx.PowerIterationFailedConvergence:
                    pass

        # Final ranking: preserve the original research implementation
        ranked = sorted(
            speaker_scores.items(),
            key=lambda x: -x[1],
        )

        ranked_names = [
            name
            for name, _ in ranked[:20]
        ]

        return ranked_names, speaker_paths
        
    
    def retrieve_graphrag(
        self,
        question,
        entities,
        graph_speakers,
        top_k_graph=15,
        top_k_vector=10,
        exclude_texts=None,
        extra_embedding=None,
    ):
        # Encode the user's question using the original embedding model.
        q_emb_raw = self.model.encode([question])

        # Optional HyDE embedding blend, used in iterative retrieval.
        if extra_embedding is not None:
            q_emb_raw = (q_emb_raw + extra_embedding) / 2.0

        query_embedding = q_emb_raw.tolist()
        exclude_texts = exclude_texts or set()
        retrieved = []

        # Stage 1: Search speeches from graph-selected speakers.
        if graph_speakers:
            try:
                graph_results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(
                        top_k_graph,
                        self.collection.count(),
                    ),
                    where={
                        "speaker": {
                            "$in": graph_speakers
                        }
                    },
                )

                for i in range(len(graph_results["documents"][0])):
                    retrieved.append(
                        {
                            "text": graph_results["documents"][0][i],
                            "speaker": graph_results["metadatas"][0][i]["speaker"],
                            "party": graph_results["metadatas"][0][i]["party"],
                            "date": graph_results["metadatas"][0][i]["date"],
                            "agenda": graph_results["metadatas"][0][i]["agenda"],
                            "distance": graph_results["distances"][0][i],
                            "source": "graph",
                        }
                    )

            except Exception as e:
                print(f"    [graph search error: {e}]")
                
        
        # Stage 2: Global vector search across all indexed speeches.
        plain_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k_vector,
        )

        for i in range(len(plain_results["documents"][0])):
            retrieved.append(
                {
                    "text": plain_results["documents"][0][i],
                    "speaker": plain_results["metadatas"][0][i]["speaker"],
                    "party": plain_results["metadatas"][0][i]["party"],
                    "date": plain_results["metadatas"][0][i]["date"],
                    "agenda": plain_results["metadatas"][0][i]["agenda"],
                    "distance": plain_results["distances"][0][i],
                    "source": "vector",
                }
            )


        # Stage 3: Retrieve speeches from matched political parties.
        for party in entities.get("parties", []):
            try:
                party_results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(15, self.collection.count()),
                    where={
                        "party": {
                            "$in": [party]
                        }
                    },
                )

                for i in range(len(party_results["documents"][0])):
                    retrieved.append(
                        {
                            "text": party_results["documents"][0][i],
                            "speaker": party_results["metadatas"][0][i]["speaker"],
                            "party": party_results["metadatas"][0][i]["party"],
                            "date": party_results["metadatas"][0][i]["date"],
                            "agenda": party_results["metadatas"][0][i]["agenda"],
                            "distance": party_results["distances"][0][i],
                            "source": "party",
                        }
                    )

            except Exception as e:
                print(f"    [party search error ({party}): {e}]")

        # Stage 4: Pinned-speaker retrieval.
        # Search specifically for speakers identified in the question.
        for named_spk in entities.get("speakers", []):
            try:
                pinned_results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(15, self.collection.count()),
                    where={
                        "speaker": {
                            "$in": [named_spk]
                        }
                    },
                )

                for i in range(len(pinned_results["documents"][0])):
                    retrieved.append(
                        {
                            "text": pinned_results["documents"][0][i],
                            "speaker": pinned_results["metadatas"][0][i]["speaker"],
                            "party": pinned_results["metadatas"][0][i]["party"],
                            "date": pinned_results["metadatas"][0][i]["date"],
                            "agenda": pinned_results["metadatas"][0][i]["agenda"],
                            "distance": pinned_results["distances"][0][i],
                            "source": "pinned",
                        }
                    )

            except Exception as e:
                print(f"    [pinned search error ({named_spk}): {e}]")

        # Stage 5: Year-stratified semantic retrieval.
        # Run additional searches for questions covering multiple years.
        year_list = sorted(entities.get("years", []))

        if len(year_list) >= 2:
            for yr in year_list:
                try:
                    # Append the year before encoding the question.
                    yr_embedding = self.model.encode(
                        [f"{question} {yr}"]
                    ).tolist()

                    # Search using the year-specific embedding.
                    yr_results = self.collection.query(
                        query_embeddings=yr_embedding,
                        n_results=5,
                    )

                    for i in range(len(yr_results["documents"][0])):
                        retrieved.append(
                            {
                                "text": yr_results["documents"][0][i],
                                "speaker": yr_results["metadatas"][0][i]["speaker"],
                                "party": yr_results["metadatas"][0][i]["party"],
                                "date": yr_results["metadatas"][0][i]["date"],
                                "agenda": yr_results["metadatas"][0][i]["agenda"],
                                "distance": yr_results["distances"][0][i],
                                "source": f"year_{yr}",
                            }
                        )

                except Exception as e:
                    print(f"    [year search error ({yr}): {e}]")

        
        # Stage 6: Keyword-text retrieval using exact phrases.
        q_lower = question.lower()
        kw_terms = []

        # Identify keyword terms associated with the question.
        for phrase, terms in _KW_TEXT_TERMS.items():
            if phrase in q_lower:
                kw_terms.extend(terms)

        # Search for documents containing each matched term.
        for term in set(kw_terms):
            try:
                kw_results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=5,
                    where_document={"$contains": term},
                )

                for i in range(len(kw_results["documents"][0])):
                    retrieved.append(
                        {
                            "text": kw_results["documents"][0][i],
                            "speaker": kw_results["metadatas"][0][i]["speaker"],
                            "party": kw_results["metadatas"][0][i]["party"],
                            "date": kw_results["metadatas"][0][i]["date"],
                            "agenda": kw_results["metadatas"][0][i]["agenda"],
                            "distance": kw_results["distances"][0][i],
                            "source": f"kw_{term[:12]}",
                        }
                    )

            except Exception:
                pass

        
        # Deduplicate by text prefix and speaker-date pair.
        # Preserve the original research implementation.
        seen_texts = {}
        seen_speaker_dates = {}

        for chunk in retrieved:
            txt = chunk["text"].strip()
            text_key = txt[:80]
            spk_date_key = (
                chunk["speaker"],
                chunk["date"],
            )

            # Skip passages excluded from an earlier retrieval pass.
            if text_key in exclude_texts:
                continue

            # If this speaker/date has already been seen,
            # retain the passage with the smaller distance.
            if spk_date_key in seen_speaker_dates:
                existing = seen_speaker_dates[spk_date_key]

                if (
                    chunk["distance"]
                    >= seen_texts[existing]["distance"]
                ):
                    continue

                del seen_texts[existing]

            # Keep new passages or closer matches for the same text.
            if (
                text_key not in seen_texts
                or chunk["distance"] < seen_texts[text_key]["distance"]
            ):
                seen_texts[text_key] = chunk
                seen_speaker_dates[spk_date_key] = text_key

        
        deduped = list(seen_texts.values())
        
        threshold = _hop_threshold(question)

        # Cross-encoder reranking with the original graph-score boost.
        if deduped:
            # Construct question-passage pairs.
            pairs = [
                [question, c["text"]]
                for c in deduped
            ]

            # Score each pair with the original cross-encoder.
            scores = self.reranker.predict(pairs)

            # Map each graph-selected speaker to their ranking position.
            graph_rank = {
                spk: i
                for i, spk in enumerate(graph_speakers)
            }

            n_graph = max(len(graph_speakers), 1)
            boosted = []

            for raw_score, chunk in zip(scores, deduped):
                spk = chunk["speaker"]

                if spk in graph_rank:
                    rank_norm = (
                        1.0 - graph_rank[spk] / n_graph
                    )

                    final_score = (
                        float(raw_score)
                        + 0.08 * rank_norm
                    )
                else:
                    final_score = float(raw_score)

                boosted.append((final_score, chunk))

            # Sort passages by their boosted relevance scores.
            ranked = sorted(
                boosted,
                key=lambda x: -x[0],
            )

            
            all_ranked_chunks = [
                dict(c, rerank_score=s)
                for s, c in ranked
            ]

            # Keep passages meeting the question-dependent threshold.
            filtered = [
                dict(c, rerank_score=s)
                for s, c in ranked
                if s >= threshold
            ]

            # Original fallback: if nothing passes, retain the top five.
            deduped = (
                filtered
                if filtered
                else all_ranked_chunks[:5]
            )
            
            # Named-speaker coverage:
            named_spks = entities.get("speakers", [])

            if named_spks:
                present = {
                    c["speaker"]
                    for c in deduped
                }

                for spk in named_spks:
                    if spk not in present:
                        best = next(
                            (
                                c
                                for c in all_ranked_chunks
                                if c["speaker"] == spk
                            ),
                            None,
                        )

                        if best:
                            deduped.append(best)
                            present.add(spk)

        
        # Final selection: enforce party diversity where applicable.
        target_parties = entities.get("parties", [])

        # Case 1: Two or more requested parties.
        if len(target_parties) >= 2:
            per_party_cap = max(
                1,
                -(-TOP_K_FINAL // len(target_parties))
            )

            party_counts = {p: 0 for p in target_parties}
            final = []
            remaining = list(deduped)

            # First, attempt to select one passage per requested party.
            for party in target_parties:
                for chunk in remaining:
                    if chunk["party"] == party:
                        final.append(chunk)
                        party_counts[party] = (
                            party_counts.get(party, 0) + 1
                        )
                        remaining.remove(chunk)
                        break

            # Fill additional places while respecting party caps.
            for chunk in remaining:
                if len(final) >= TOP_K_FINAL:
                    break

                p = chunk["party"]

                if (
                    p in party_counts
                    and party_counts[p] >= per_party_cap
                ):
                    continue

                final.append(chunk)
                party_counts[p] = party_counts.get(p, 0) + 1

            # Fill any positions still available.
            for chunk in remaining:
                if len(final) >= TOP_K_FINAL:
                    break

                if chunk not in final:
                    final.append(chunk)

            final.sort(
                key=lambda c: c.get("rerank_score", 0),
                reverse=True,
            )

            return _post_process(
                final[:TOP_K_FINAL],
                question,
                deduped,
                TOP_K_FINAL,
            )

        # Case 2: Exactly one requested party.
        elif len(target_parties) == 1:
            target_party = target_parties[0]

            min_target = max(
                1,
                -(-TOP_K_FINAL // 2)
            )

            final = []
            remaining = list(deduped)

            # Prioritise passages from the requested party.
            for chunk in remaining[:]:
                if len(final) >= min_target:
                    break

                if chunk["party"] == target_party:
                    final.append(chunk)
                    remaining.remove(chunk)

            # Fill the remaining positions by relevance.
            for chunk in remaining:
                if len(final) >= TOP_K_FINAL:
                    break

                final.append(chunk)

            final.sort(
                key=lambda c: c.get("rerank_score", 0),
                reverse=True,
            )

            return _post_process(
                final[:TOP_K_FINAL],
                question,
                deduped,
                TOP_K_FINAL,
            )

        # Case 3: No requested parties.
        return _post_process(
            deduped[:TOP_K_FINAL],
            question,
            deduped,
            TOP_K_FINAL,
        )
        
    def generate_interim(self, question, context_chunks):
        # Build the original interim-answer context.
        context = ""

        for i, chunk in enumerate(context_chunks[:3]):
            context += (
                f"\n--- Speech {i+1} "
                f"({chunk['speaker']}, {chunk['party']}, "
                f"{chunk['date']}) ---\n"
            )
            context += chunk["text"] + "\n"

        # Preserve the original three-attempt retry behaviour.
        for attempt in range(3):
            try:
                resp = self.llm_client.chat.completions.create(
                    model=self.ollama_model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Based on these parliamentary speeches, briefly answer:\n\n"
                                f"{context}\n\n"
                                f"QUESTION: {question}\n\n"
                                f"BRIEF ANSWER (2-3 sentences, name specific MPs, parties, topics, years):"
                            ),
                        }
                    ],
                    temperature=0.1,
                    max_tokens=150,
                )

                return resp.choices[0].message.content or ""

            except Exception:
                if attempt < 2:
                    time.sleep(5)
                else:
                    return ""
    
    def merge_retrieval_passes(
        self,
        question,
        entities,
        chunks_r1,
        chunks_r2,
        top_k_final=5,
    ):
        # Exclude first-pass passages from the R2 candidates.
        r1_texts = {
            c["text"].strip()[:80]
            for c in chunks_r1
        }

        combined = list(chunks_r1) + [
            c for c in chunks_r2
            if c["text"].strip()[:80] not in r1_texts
        ]

        # Preserve the original relevance ranking.
        combined.sort(
            key=lambda c: c.get("rerank_score", 0),
            reverse=True,
        )

        # Use parties from the original question.
        target_parties = entities.get("parties", [])

        if target_parties:
            per_cap = max(
                1,
                -(-top_k_final // max(1, len(target_parties))),
            )

            selected = []
            party_counts = {}

            # Include evidence from each requested party.
            for party in target_parties:
                for c in combined:
                    if c["party"] == party and c not in selected:
                        selected.append(c)
                        party_counts[party] = 1
                        break

            # Fill positions subject to the party cap.
            for c in combined:
                if len(selected) >= top_k_final:
                    break

                if c in selected:
                    continue

                party = c["party"]

                if (
                    party in party_counts
                    and party_counts[party] >= per_cap
                ):
                    continue

                selected.append(c)
                party_counts[party] = (
                    party_counts.get(party, 0) + 1
                )

            # Fill any remaining positions.
            for c in combined:
                if len(selected) >= top_k_final:
                    break

                if c not in selected:
                    selected.append(c)

            selected.sort(
                key=lambda c: c.get("rerank_score", 0),
                reverse=True,
            )

            final_chunks = _post_process(
                selected[:top_k_final],
                question,
                combined,
                top_k_final,
            )

        else:
            final_chunks = _post_process(
                combined[:top_k_final],
                question,
                combined,
                top_k_final,
            )

        return final_chunks

    def retrieve_iterative(self, question):
        # Stage 1: Extract entities from the original question.
        entities = self.extract_entities(question)

        # Preserve the original definitional-query behaviour.
        is_definitional = bool(
            re.match(r"^what is\b", question.strip().lower())
        )

        # Stage 2: Initial graph traversal.
        if is_definitional:
            graph_speakers = []
            speaker_paths = {}
        else:
            graph_speakers, speaker_paths = self.traverse_graph(
                entities,
                question,
            )
            
        # Build initial speaker profiles from graph evidence.
        speaker_profiles = self.build_speaker_profiles(
            graph_speakers,
            entities,
        )

        # Stage 3: First retrieval pass.
        chunks_r1 = self.retrieve_graphrag(
            question,
            entities,
            graph_speakers,
        )

        final_chunks = chunks_r1
        chunks_r2 = []
        merged_entities = entities
        graph_speakers_r2 = []
        interim = ""
        r2_count = 0

        # The original pipeline skips R2 for definitional
        # questions or when R1 returns no evidence.
        if chunks_r1 and not is_definitional:

            # Stage 4: Generate an interim answer.
            try:
                interim = self.generate_interim(
                    question,
                    chunks_r1,
                )
            except Exception:
                interim = ""

            # Stage 5: Expand entities.
            new_entities = self.extract_entities(interim)

            merged_entities = merge_entities(
                entities,
                new_entities,
            )

            # Repeat graph traversal with expanded entities.
            graph_speakers_r2, paths_r2 = self.traverse_graph(
                merged_entities,
                question,
            )

            speaker_paths.update(paths_r2)
            
            # Incorporate profiles based on expanded entities.
            speaker_profiles.update(
                self.build_speaker_profiles(
                    graph_speakers_r2,
                    merged_entities,
                )
            )

            # Stage 6: Prepare HyDE embedding.
            hyde_embedding = None

            if interim:
                hyde_embedding = self.model.encode([interim])

            # Exclude passages selected in R1.
            r1_texts = {
                c["text"].strip()[:80]
                for c in chunks_r1
            }

            # Second retrieval pass.
            chunks_r2 = self.retrieve_graphrag(
                question,
                merged_entities,
                graph_speakers_r2,
                exclude_texts=r1_texts,
                extra_embedding=hyde_embedding,
            )

            # Stage 7: Merge and select final evidence.
            final_chunks = self.merge_retrieval_passes(
                question,
                entities,
                chunks_r1,
                chunks_r2,
            )

            r2_count = sum(
                c not in chunks_r1
                for c in final_chunks
            )

        # Return the evidence and useful diagnostics.
        return {
            "question": question,
            "entities": entities,
            "merged_entities": merged_entities,
            "graph_speakers": graph_speakers,
            "graph_speakers_r2": graph_speakers_r2,
            "speaker_paths": speaker_paths,
            "chunks_r1": chunks_r1,
            "chunks_r2": chunks_r2,
            "final_chunks": final_chunks,
            "interim_answer": interim,
            "r2_count": r2_count,
            "speaker_profiles": speaker_profiles,
        }

    def build_speaker_profiles(self, graph_speakers, entities):
        profiles = {}

        topic_sims = entities.get("topic_sims", {})

        # Select up to five of the most relevant graph topics.
        relevant_topics = sorted(
            topic_sims,
            key=lambda t: -topic_sims[t],
        )[:5]

        for speaker in graph_speakers:
            if not self.graph.has_node(speaker):
                continue

            party = next(
                (
                    nb
                    for nb in self.graph.neighbors(speaker)
                    if self.graph.nodes[nb].get("node_type") == "PARTY"
                ),
                "Unknown",
            )

            total_speeches = self.graph.nodes[speaker].get(
                "speech_count", 0
            )

            topic_stances = []

            for topic in relevant_topics:
                if not self.graph.has_edge(speaker, topic):
                    continue

                edge = self.graph[speaker][topic]
                cnt = edge.get("speech_count", 0)

                if cnt == 0:
                    continue

                zscores = self._topic_sentiment_zscores(topic)
                z = zscores.get(speaker, 0)

                stance = (
                    "supportive" if z > 0.5
                    else "critical" if z < -0.5
                    else "neutral"
                )

                topic_stances.append(
                    f"{topic} ({cnt} speeches, {stance})"
                )

            if topic_stances:
                profiles[speaker] = {
                    "party": party,
                    "total_speeches": total_speeches,
                    "stances": topic_stances,
                }

        return profiles
    
    def generate(
        self,
        question,
        context_chunks,
        speaker_profiles,
        speaker_paths,
    ):
        # Place highly ranked evidence near the beginning
        # and end of the context, as in the original pipeline.
        context_chunks = _reorder_lost_in_middle(
            context_chunks
        )

        context = ""

        for i, chunk in enumerate(context_chunks):
            spk = chunk["speaker"]
            path = speaker_paths.get(
                spk,
                "plain vector search",
            )

            context += (
                f"\n--- Speech {i+1} "
                f"({spk}, {chunk['party']}, {chunk['date']}) ---\n"
                f"[Retrieved via: {path}]\n"
            )

            context += chunk["text"] + "\n"

        # Add graph-derived speaker profiles to the prompt.
        profiles_text = ""

        for speaker in {
            c["speaker"] for c in context_chunks
        }:
            if speaker in speaker_profiles:
                p = speaker_profiles[speaker]

                stances = (
                    "; ".join(p["stances"])
                    if p["stances"]
                    else "general Brexit debates"
                )

                profiles_text += (
                    f"  {speaker} ({p['party']}, "
                    f"{p['total_speeches']} total speeches): "
                    f"{stances}\n"
                )

        profiles_section = (
            f"\nSPEAKER PROFILES (from parliamentary record):\n"
            f"{profiles_text}"
            if profiles_text else ""
        )

        # Select instructions according to question type.
        q = question.lower()

        if any(
            w in q
            for w in [
                "differ", "compare", "contrast",
                "versus", " vs ", "between",
            ]
        ):
            task_instruction = (
                "This is a COMPARISON question. Present each "
                "party's or speaker's position as a clearly "
                "labelled separate paragraph before drawing "
                "any contrast."
            )

        elif any(
            w in q
            for w in [
                "evolve", "evolution", "change",
                "over time", "from ", "to 20",
            ]
        ):
            task_instruction = (
                "This is a TEMPORAL EVOLUTION question. "
                "Organise your answer chronologically, "
                "noting how positions or debates shifted "
                "across the years covered by the speeches."
            )

        elif any(
            w in q
            for w in [
                "why", "reason", "cause",
                "led to", "result",
            ]
        ):
            task_instruction = (
                "This is a CAUSAL question. Identify "
                "specific reasons, events, or arguments "
                "from the speeches that explain the outcome."
            )

        else:
            task_instruction = (
                "Answer comprehensively, citing specific "
                "speakers, dates, and stances."
            )

        # Construct the original research prompt.
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

        # Generate the final answer using the original model
        # and generation parameters.
        for attempt in range(3):
            try:
                response = (
                    self.llm_client.chat.completions.create(
                        model=self.ollama_model,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        temperature=0.1,
                        max_tokens=768,
                    )
                )

                return (
                    response.choices[0].message.content
                    or "ERROR: No response from LLM"
                )

            except Exception as e:
                if attempt < 2:
                    print(
                        f"  [generate retry {attempt+1}: "
                        f"{type(e).__name__}]"
                    )
                    time.sleep(5)
                else:
                    return (
                        "ERROR: LLM unavailable after retries"
                    )









        
        
        