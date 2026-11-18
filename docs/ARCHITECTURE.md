
# ParliamentAI Architecture

## 1. Overview

ParliamentAI is a full-stack research application built on a graph-guided retrieval system for UK parliamentary Brexit speeches.

It has two main layers:

1. An offline research and indexing workflow that prepares parliamentary speeches, embeddings and a knowledge graph.
2. An interactive question-answering workflow that retrieves relevant evidence and presents grounded answers through FastAPI and React.

The original offline experiment compares a hybrid RAG baseline against GraphRAG. The current interactive service is a subsequently developed retrieval and answer-generation implementation.

## 2. Offline data pipeline

```text
ParlSpeech V2 archive
       |
       v
Brexit keyword and date filtering
       |
       v
Speaker normalisation and deduplication
       |
       v
Filtered parliamentary speeches
       |
       +---------------------+
       |                     |
       v                     v
Sentence-aware chunking   Knowledge graph construction
       |                     |
       v                     v
Sentence embeddings       NetworkX speaker-party-
       |                  topic-year graph
       v                     |
ChromaDB collection          |
       |                     |
       +----------+----------+
                  |
                  v
         GraphRAG retrieval
```

The source data-loading script filters records from 2015-01-01 through 2020-12-31.

The full generated data artifacts are intentionally excluded from the public Git repository.

## 3. Knowledge graph

The graph is a NetworkX undirected graph with four main node types:

| Node | Description |
|---|---|
| SPEAKER | Parliamentary speaker |
| PARTY | Recorded political party |
| TOPIC | Normalised parliamentary agenda topic |
| YEAR | Year of discussion |

Its principal relationship categories are:

| Relationship | Interpretation |
|---|---|
| MEMBER_OF | Speaker associated with a party |
| SPOKE_ABOUT | Speaker discussed a topic |
| DISCUSSED_IN | Topic appeared in a year |
| DEBATED_WITH | Speakers participated in the same debates repeatedly |

The DEBATED_WITH relation is a co-participation signal. It does not prove that two MPs directly replied to or argued with one another.

Graph edges retain aggregate metadata such as speech frequency and sentiment polarity.

TextBlob sentiment is a generic language-polarity feature, not a reliable political-stance classifier.

## 4. Embeddings and document retrieval

The vectorisation stage divides speeches into overlapping sentence-aware passages, targeting approximately 200 words per chunk with overlap.

Each indexed passage retains identifying metadata including:

- Speaker
- Party
- Speech date
- Parliamentary agenda
- Source speech index

Passages are embedded using BAAI/bge-small-en-v1.5 and stored in ChromaDB.

The original research comparison also uses BM25, HyDE, reciprocal-rank fusion and cross-encoder reranking as part of its hybrid retrieval design.

## 5. Interactive question-answering pipeline

```text
User question
     |
     v
React / ParliamentAI
     |
     v
POST /graphrag/ask
     |
     v
FastAPI route
     |
     v
GraphRAG service
     |
     +-- Question classification and entity cues
     |
     +-- Question-aware or temporal retrieval
     |
     +-- Graph-guided candidate selection
     |
     +-- ChromaDB passage retrieval
     |
     +-- Candidate reranking and evidence selection
     |
     v
Grounded answer generation
     |
     v
Answer and evidence payload
     |
     v
React rendering and research tools
```

The active application uses the optimised GraphRAG engine and dedicated evidence-selection and grounded-generation modules.

Explicit multi-year questions receive additional year-aware retrieval handling. Other questions use question-aware retrieval designed to cover the central facets of a question.

The default local application configuration disables the optional second retrieval pass and uses extractive grounded generation.

## 6. FastAPI interface

The React application calls:

`POST /graphrag/ask`

Request body:

```json
{
  "question": "How did Labour and Conservative Brexit positions differ?"
}
```

The response includes fields with the following general structure:

```json
{
  "question": "...",
  "answer": "... [E1] ...",
  "evidence": [
    {
      "speaker": "...",
      "party": "...",
      "date": "...",
      "agenda": "...",
      "text": "...",
      "source": "...",
      "rerank_score": 0.0
    }
  ],
  "retrieval": {},
  "coverage": {}
}
```

This is an illustrative contract, not a recorded answer.

Evidence references such as `[E1]` refer to one-based positions in the evidence array. The frontend presents the corresponding original passage in its source panel.

The legacy `/ask` endpoint returns a placeholder response and is not the main application route.

## 7. Frontend architecture

The frontend is a React application built with Vite.

Its main user-facing components include:

- Research question composer
- Conversation view
- Evidence source panel
- Party and year filters
- Citation navigation
- Searchable conversation history
- Copy and export actions

The application stores saved conversations and their evidence in browser IndexedDB.

The selected API endpoint is stored separately in browser localStorage.

Markdown and JSON exports are generated client-side. Conversation histories are not uploaded to a separate database by these features.

## 8. Evidence integrity

The answer generator is designed to use retrieved evidence rather than present unsupported historical synthesis.

Nevertheless, evidence grounding has two separate requirements:

1. Structural correctness: evidence is present and cited markers refer to valid passages.
2. Interpretative correctness: a summary accurately represents the speaker, context and meaning of those passages.

The application's automated checks cover structural behaviour more thoroughly than independent historical validation.

In particular, mentioning an opposing argument is not equivalent to endorsing it. This distinction remains important for political-stance questions.

## 9. Evaluation

The original offline benchmark compares hybrid RAG and GraphRAG over 50 questions using retrieval, similarity, faithfulness, precision and diversity measures.

It is distinct from the eight-question structural acceptance suite used during the interactive service's development.

The current FastAPI contract tests mock the retrieval service to isolate routing and response-format behaviour.

See `results/RESULTS.md` and `docs/METHODOLOGY.md`.

## 10. Deployment considerations

The working application was verified locally.

A public deployment would require additional work, including:

- Secure backend and frontend hosting
- Dependency advisory review
- Reproducible access to the required model/data artifacts
- Resource and startup management
- Request limits and operational monitoring
- Broader factual and adversarial testing

Local functional verification should not be interpreted as completed production hardening.
