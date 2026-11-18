# ParliamentAI local Docker setup (first-pass development configuration)

Docker Compose runs the original FastAPI application and the React build without modifying their active source code. It runs only in extractive mode, without requiring Ollama. The vector database must be copied into a separate directory before the first run.

## Safety preflight

- Make sure Docker Desktop is running and Linux containers are selected.
- Confirm at least 6 GB available on the drive containing the repository for the index copy, and additional space for Docker images and model cache (often on C:).
- STOP the existing local FastAPI process before copying ChromaDB, to avoid a partially consistent SQLite copy.
- The original graph/index remain in `data/`; the copy lives in `data/docker_runtime/` (ignored by Git via `/data/*`).

## Setup from repository root

Run `docker compose config` to validate the Compose file.

```powershell
.\scripts\prepare_docker_data.ps1

docker compose build

docker compose up -d

docker compose ps
```

Open http://localhost:8080 and first check http://localhost:8080/backend/health.

The first real GraphRAG question can download two ML models into the persistent `model_cache` Docker volume, so it may be slower and require additional storage. This workflow has not yet been tested on the user's machine.

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
