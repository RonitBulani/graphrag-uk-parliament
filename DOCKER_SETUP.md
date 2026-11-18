# ParliamentAI local Docker setup

Docker Compose runs the original FastAPI application and the React build without modifying their active source code. It runs only in extractive mode, without requiring Ollama. The vector database must be copied into a separate directory before the first run.

## Safety preflight

- Make sure Docker Desktop is running and Linux containers are selected.
- Confirm at least 6 GB available on the drive containing the repository for the index copy, and additional space wherever Docker Desktop stores images and volumes.
- STOP the existing local FastAPI process before copying ChromaDB, to avoid a partially consistent SQLite copy.
- The original graph/index remain in `data/`; the copy lives in `data/docker_runtime/` (ignored by Git via `/data/*`).

## Setup from repository root

Run `docker compose config` to validate the Compose file. Prepare the separate runtime data copy only once: the preparation script deliberately refuses to overwrite an existing `data/docker_runtime` directory.

```powershell
# Only when data/docker_runtime does not already exist:
.\scripts\prepare_docker_data.ps1

docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
```

Open http://localhost:8080 and first check http://localhost:8080/backend/health.

The first real GraphRAG question can download two ML models into the persistent `model_cache` Docker volume, so it may be slower and require additional storage.

**Local validation (November 2026):** On Windows with Docker Desktop and the WSL2 backend, both images built successfully, the frontend loaded, FastAPI responded to health checks, real GraphRAG POST requests returned HTTP 200, and the citation/source-inspection interface worked. This does not constitute independent verification of every historical claim. On a simultaneous `docker compose restart`, Nginx may temporarily return HTTP 502 until FastAPI is healthy; check `docker compose ps` and retry after the backend is ready.

## Troubleshooting

```powershell
docker compose logs --tail=150 backend
docker compose logs --tail=100 frontend
docker compose ps
docker stats
```

Stop without removing data:

```powershell
docker compose down
```

DO NOT run `docker compose down -v`: that would remove the cached model volume. Do not run `docker system prune --volumes` while you want to preserve caches.

## Deployment limitations

This is a local development deployment, **not** a public production configuration. Before public hosting, add HTTPS, rate limiting, host/security headers, health probes that exercise retrieval, resource limits, backup strategy, and evaluate dataset licensing/privacy requirements. The Compose port is bound to `127.0.0.1` and is not publicly exposed.
