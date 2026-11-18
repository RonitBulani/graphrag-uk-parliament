
# Reproducibility Guide

This document explains how to recreate the ParliamentAI environment and the original GraphRAG research experiment.

## 1. Requirements

The project was developed and tested locally using:

- Python 3.12.16
- FastAPI 0.162.0
- ChromaDB 1.5.8
- NetworkX 3.6.1
- sentence-transformers 5.4.1
- React 19
- Vite 8
- Node.js and npm
- Windows PowerShell

The Python package versions are listed in `requirements.txt`. JavaScript dependency versions are controlled by `frontend/package-lock.json`.

The complete dependency set has not yet been independently verified in a fresh environment.

## 2. Clone the repository

```powershell
git clone https://github.com/RonitBulani/graphrag-uk-parliament.git
cd graphrag-uk-parliament
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

If pip is missing from a new virtual environment, check its Python installation and run:

```powershell
python -m ensurepip --upgrade
python -m pip --version
```

Check installed dependencies:

```powershell
python -m pip check
```

Do not blindly upgrade the entire working environment to resolve a single packaging warning.

## 3. Data availability

The research source is **ParlSpeech V2** (Rauh and Schwalbach, 2020; [Harvard Dataverse, DOI:10.7910/DVN/L4OAKN](https://doi.org/10.7910/DVN/L4OAKN)). The released UK data are supplied as `Corp_HouseOfCommons_V2.rds`, whereas this repository's preprocessing script expects a separately prepared CSV ZIP archive. The code in this repository does **not** convert the official RDS download into that ZIP. See [data provenance and input contract](../data/README.md).

The preprocessing script is configured with:

```text
DATE_START = 2015-01-01
DATE_END   = 2020-12-31
```

These are inclusive filtering bounds, not a claim of 2020 coverage in the ParlSpeech V2 release. The UK source extends to 2019. The current interactive service enforces a separate 2015–2019 supported-question range.

The public repository contains:

- `data/questions.json`: the original 50-question benchmark
- `data/parlspeech_brexit_sample_50.csv`: sample speech records
- `data/README.md`: data-source and rebuilding instructions

It does not contain:

- The source archive
- The full filtered speech dataset
- The complete ChromaDB vector database
- The generated NetworkX GraphML file
- Local Ollama model weights
- Large generated answer dumps and development traces

A small sample alone is not sufficient to recreate the complete application's retrieval behaviour.

## 4. Prepare source data

Obtain and cite the official ParlSpeech V2 UK source dataset (linked above). **Do not rename the official `.rds` file to `.zip`: the formats are different.** Before running the existing preprocessing script, prepare a separate ZIP file named `archive.zip` at the repository root. Its internal entries must be exactly:

```text
df_HoC_2000s.csv
df_HoC_miniDebate.csv
```

Each CSV must provide at least `date`, `agenda`, `speechnumber`, `speaker`, `party`, and `text`; the script reads the two CSVs directly from the ZIP, applies its configured keyword/date filters and writes the derived data. **The transformation that produced these specially named CSVs is not supplied here**, so an independent reproduction starting only with the official RDS file is not yet end-to-end. Use the prepared archive only when its provenance and schema have been checked.

Run from the repository root:

```powershell
python src/01_data_loading.py
```

This produces:

```text
data/parlspeech_brexit_filtered.csv
data/parlspeech_brexit_sample_50.csv
```

The sample CSV is generated with a fixed random seed for inspection.

## 5. Build the retrieval artifacts

The canonical indexing script is:

```text
src/02_vectorisation.py
```

It creates sentence-aware overlapping chunks, embeds them using `BAAI/bge-small-en-v1.5`, and writes a persistent ChromaDB collection.

The script rebuilds the collection from scratch. Back up an existing index before running it.

```powershell
python src/02_vectorisation.py
```

Build the knowledge graph with:

```powershell
python src/03_knowledge_graph.py
```

This writes the generated graph and metadata to the paths configured in the script.

These operations can require substantial processing time, memory and local storage.

## 6. Prepare local models

The original evaluation pipeline uses locally served LLMs through Ollama.

The documented experiment models include:

```powershell
ollama pull qwen2.5:3b
ollama pull phi3:mini
```

Start Ollama according to its local installation instructions.

The current extractive application mode does not require an LLM generation call for every answer, but some alternative modes and the original experiments depend on local model availability.

## 7. Start the FastAPI backend

From the repository root, with the Python virtual environment activated:

```powershell
$env:GRAPHRAG_ENABLE_R2 = "0"
$env:GRAPHRAG_GENERATION_MODE = "extractive"
$env:GRAPHRAG_NLI_ENABLED = "0"

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

API documentation:

`http://127.0.0.1:8000/docs`

Main application route:

`POST /graphrag/ask`

The earlier `/ask` route is a placeholder and is not the frontend's default.

The backend requires the correct full retrieval artifacts at the configured local paths.

## 8. Start React

Open a second PowerShell terminal:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

Open:

`http://127.0.0.1:5173/`

The Vite development server proxies `/backend` requests to FastAPI at port 8000.

The source evidence panel, filters, history and export features are available after the API returns evidence.

## 9. Frontend quality checks

```powershell
cd frontend

npm.cmd run lint
npm.cmd run build
npm.cmd audit --omit=dev
```

During release preparation, lint and build both passed.

The production-dependency audit reported zero vulnerabilities. A separate development-dependency advisory involving `picomatch` remained unresolved.

The Vite development server should not be exposed directly as a public production deployment.

## 10. API contract tests

From the repository root:

```powershell
python -m pytest -q tests/test_main.py
```

The current five tests cover:

1. The health endpoint
2. Registration of the `/graphrag/ask` POST route
3. The expected answer-and-evidence response contract using a mocked service
4. Rejection of an empty or whitespace-only question
5. Trimming surrounding whitespace before passing a question to the service

These tests passed in the working environment.

They do not independently measure retrieval quality or historical factual correctness.

The application imports can also have nontrivial startup overhead.

## 11. Original research pipeline

After rebuilding the corpus, vector store and graph, the original experiment can be executed with:

```powershell
python src/04_naive_rag.py
python src/05_graphrag.py
python src/06_generate_pooled_references.py
python src/07_evaluation.py
```

Figures can be recreated using:

```powershell
python scripts/make_figures.py
```

The offline pipeline uses local models, retrieval artifacts and generated reference data.

Evaluation can be computationally expensive. Do not assume the complete pipeline is a quick integration test.

## 12. Evaluation interpretation

The original 50-question benchmark evaluates both systems across 1-hop, 2-hop and 3-hop questions.

MRR and nDCG@5 use target speaker and party sets derived from pooled evaluation metadata.

These are entity-based relevance proxies, not independently human-annotated passage judgements.

Pooled reference answers are generated from evidence retrieved by both systems. They are not independently verified historical ground truth.

Because generated references and model outputs can vary, an exact reproduction may depend on environment, model versions and randomisation settings.

## 13. Conversation storage and exports

ParliamentAI saves conversations locally using browser IndexedDB.

The export menu supports:

- Current conversation as Markdown
- Current conversation as JSON
- All saved conversations as JSON

Exported records include full retrieved evidence.

The backup functionality does not currently support importing records into another browser.

Clearing browser storage can remove local conversation history.

## 14. Known limitations

- Full data and generated indexes must be recreated or supplied separately.
- The project has been verified locally on Windows, including a two-container Docker Compose build and functioning GraphRAG requests. See [Docker setup](../DOCKER_SETUP.md).
- Structural citation checks do not guarantee historical correctness.
- Speaker co-participation does not imply direct debate interaction.
- TextBlob polarity is not a political-stance label.
- Broader answer-quality evaluation is still needed.
- Public hosting and production security hardening are not complete.

## 15. Suggested verification order

For a fresh environment:

1. Install dependencies and check package compatibility.
2. Confirm the full source data and generated retrieval artifacts.
3. Verify the FastAPI health and question endpoints.
4. Run API contract tests.
5. Start React and confirm a real answer with evidence.
6. Test citation navigation and evidence filters.
7. Verify history persistence and exports.
8. Run frontend lint and production build.

The original research results are documented separately in `results/RESULTS.md`.
