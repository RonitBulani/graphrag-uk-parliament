# ParliamentAI demonstration guide

This guide explains how to record **authentic screenshots and a short walkthrough** of the running local application. No fabricated answers or example screenshots are included in the repository.

## Prerequisites

- Working FastAPI backend with the complete local graph and ChromaDB collection
- Working React frontend
- Full parliamentary data/index prepared locally, as explained in [Reproducibility](REPRODUCIBILITY.md)
- No confidential local paths, API keys, browser account details or unrelated tabs visible during recording

From the repository root, start the backend in one PowerShell terminal:

```powershell
$env:GRAPHRAG_ENABLE_R2 = "0"
$env:GRAPHRAG_GENERATION_MODE = "extractive"
$env:GRAPHRAG_NLI_ENABLED = "0"

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in a second PowerShell terminal:

```powershell
cd frontend
npm.cmd run dev
```

Open `http://127.0.0.1:5173/` and confirm the backend is available.

## Recommended real screenshots

Save screenshots inside `docs/screenshots/` once they have been captured from the running application. Suggested filenames:

| File | What to show |
|---|---|
| `01-research-workspace.png` | Clean main interface and question composer |
| `02-grounded-answer.png` | Completed answer with numbered evidence citations |
| `03-source-inspection.png` | A selected citation and the corresponding parliamentary source passage |
| `04-evidence-filters.png` | Party/year evidence filters applied to a completed answer |
| `05-history-and-export.png` | Saved conversation history and the export interface |

Use a browser window at approximately 1440 × 900 for desktop screenshots. A second narrow viewport can demonstrate responsive layout if desired.

Avoid sharing private browser information. Do not include fake parliamentary answers or replace the source metadata with invented content. Historical interpretations should be verified against the quoted evidence before publication.

## Suggested 75–90 second walkthrough

1. **Opening (0–10 seconds).** Show the ParliamentAI research workspace and identify the dataset as UK parliamentary Brexit debates.
2. **Question (10–25 seconds).** Ask: “How did the customs union debate evolve from 2016 to 2019?” Allow the real backend to return an answer.
3. **Evidence (25–45 seconds).** Click a numbered citation such as `[E1]` and show the matching speaker, party, date and passage. Explain that the quotation can be inspected rather than accepting the summary blindly.
4. **Filtering (45–60 seconds).** Demonstrate a year or party filter. Explain that filtering changes which evidence is shown; it does not independently verify historical correctness.
5. **Research workflow (60–75 seconds).** Open an existing saved conversation, then show Markdown or JSON export.
6. **Closing (75–90 seconds).** Briefly mention the Python/FastAPI retrieval system, ChromaDB, NetworkX and React. State that the interface is a local prototype, not a hosted public chatbot.

Record with a screen recorder such as OBS Studio, or the operating system's built-in recording tool. Export an MP4 file and check that on-screen text is legible without zooming.

## Publishing the demo

- Keep a short, compressed demo file in a GitHub release or an externally hosted video link rather than committing a large MP4 to Git.
- Add links to the README only after the actual files or video are available.
- If screenshots are added, use meaningful alt text in the README.
- Preserve the original research benchmark separately from the interactive application demonstration.

## Accuracy note

The 50-question hybrid RAG vs GraphRAG comparison is the original offline experiment. Screenshots and demo questions show the current application and should **not** be presented as additional benchmark results.

The current GitHub Actions checks validate API routing and frontend packaging; they do not run retrieval over the full corpus or establish historical factual accuracy.
