# ParliamentAI local demonstration guide

ParliamentAI is currently a **local-first portfolio research application**, not a public website. Its five screenshots were captured from the running application and are available in the [screenshot gallery](SCREENSHOT_GALLERY.md).

## Starting the local Docker demonstration

The complete ChromaDB vector store, knowledge graph and metadata must already be available in `data/docker_runtime/`; they are not included in GitHub. Follow [Docker setup](../DOCKER_SETUP.md) for the one-time safe data copy.

From the repository root:

```powershell
docker compose up -d
docker compose ps
```

After the backend is healthy, open `http://localhost:8080`. Check `http://localhost:8080/backend/health` if the frontend cannot connect. The first actual research request may initialise models and take longer.

## Evidence-backed walkthrough

1. **Research workspace.** Show suggested questions, the composer and searchable conversation history.
2. **Ask a question.** For example: “How did the customs union debate evolve from 2016 to 2019?”
3. **Inspect evidence.** Open a numbered citation like `[E1]` to inspect the associated parliamentary passage and speaker/date metadata.
4. **Filter the evidence.** Filter the already retrieved passages by party, year or cited-only status. These controls filter the displayed sources; they do **not** rerun retrieval with a new constraint.
5. **Reuse the research.** Open a saved conversation and export it as Markdown or JSON, or export all conversations as a JSON backup.

## Published screenshot gallery

| Screenshot | Demonstrated feature |
|---|---|
| [Research workspace](screenshots/01-research-workspace.png) | Question composer and conversations |
| [Grounded answer](screenshots/02-grounded-answer.png) | Time-organised answer with evidence markers |
| [Source inspection](screenshots/03-source-inspection.png) | Citation navigation and passage highlighting |
| [Conversation history](screenshots/04-conversation-history.png) | Browser-local saved conversations |
| [Research exports](screenshots/05-research-export.png) | Markdown, JSON and conversation backup |

These are authentic local application screenshots, not benchmark results. A citation identifies retrieved evidence; it is not independent proof of the historical interpretation. The 50-question hybrid RAG versus GraphRAG benchmark belongs to the original research and should not be conflated with UI demonstrations.

No video recording is required for the local portfolio release.
