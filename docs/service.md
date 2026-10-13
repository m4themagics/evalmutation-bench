# Evaluation service

The optional service exposes durable benchmark runs through a typed REST API.
The CLI and worker share the same bounded async execution engine. Benchmark
contracts and scoring remain usable without a web server or database.

```mermaid
flowchart LR
  Client -->|Bearer token + Idempotency-Key| API[FastAPI]
  API -->|transaction: enqueue or return existing run| DB[(PostgreSQL)]
  Worker -->|atomic claim with lease| DB
  Worker -->|bounded async calls| Judge[Evaluator / model endpoint]
  Worker -->|predictions + manifest + metrics| DB
  API -->|GET run status and result| Client
```

## Start with PostgreSQL

Python 3.12, Docker Compose, and uv are required. Supply secrets through your
environment or a local `.env` file (ignored by Git and excluded from the image).
Use a URL-safe database password for this Compose example. Configure a separate
token of at least 32 characters. These examples generate values locally:

```bash
export EVALMUT_DB_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export EVALMUT_SERVICE_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose up --build -d
curl http://127.0.0.1:8000/health/ready
curl -X POST http://127.0.0.1:8000/v1/runs \
  -H "Authorization: Bearer $EVALMUT_SERVICE_TOKEN" \
  -H 'Idempotency-Key: validation-deterministic-001' \
  -H 'Content-Type: application/json' \
  -d '{"split":"validation","adapter":"deterministic"}'
# Replace RUN_ID with the returned UUID:
curl http://127.0.0.1:8000/v1/runs/RUN_ID \
  -H "Authorization: Bearer $EVALMUT_SERVICE_TOKEN"
```

`init` creates the initial schema explicitly. The API never issues DDL on startup.
The worker runs in a separate process; accepted jobs survive API restarts.
Compose binds the API to localhost, keeps the database on the internal network,
persists database data in a named volume, and runs the application as a non-root
user. Use `docker compose stop` to stop processes while preserving results.

## Local development

```bash
uv sync --extra dev --extra service
export EVALMUT_DATABASE_URL='sqlite+aiosqlite:///artifacts/service.db'
mkdir -p artifacts
export EVALMUT_SERVICE_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run evalmut-db
uv run uvicorn evalmutationbench.service.api:create_app --factory
# In a second shell with the same environment:
uv run evalmut-worker
```

SQLite is a development fallback. CI runs claim/idempotency tests against both
SQLite and a real PostgreSQL instance. Each store operation opens an independent
transaction; no async SQLAlchemy session is shared between tasks.

## API contract and failure behavior

- `POST /v1/runs`: `202`, a durable run UUID, and an immutable request record.
- Repeating the same key and request returns the existing run. A changed request
  with the same key returns `409`, preventing duplicate submissions on retries.
- `GET /v1/runs/{UUID}` returns status, result, and a sanitized error code.
- Only installed, operator-approved adapters and named public splits are accepted.
  Clients cannot supply paths, Python import specifications, endpoint URLs, or
  provider API keys. Operator secrets are environment variables.
- Queue ownership is an atomic SQL update with an expiring claim token. A stale
  worker cannot overwrite a completed or expired run. Multiple workers can poll
  the same database; each owns at most one run at a time.
- A run has a total deadline; its lease includes a 30-second completion margin.
  Expired runs are marked failed on the next worker poll. Provider calls are not
  automatically retried: timeout/cancellation may still incur provider charges.
- Pair errors produce a failed run with inspectable predictions and no score.
  Scoring and bootstrap reject errors, missing outputs, and duplicate IDs.
- Challenge runs return predictions without metrics or hidden labels.
- The worker reads one byte snapshot and checks the submitted SHA-256 before
  parsing it. Result manifests record the dataset hash and execution settings.
- Raw provider exception strings are redacted. Logs contain run IDs.

Native LLM adapters use LiteLLM `acompletion`. Synchronous plugins use the thread
compatibility path and must provide their own I/O deadlines and thread safety;
cancelling an await does not kill a Python thread. Concurrency controls bound
active async evaluations to 1–32 workers and preserve dataset order. A timed-out
sync thread may continue running until its own deadline. Process-level
parallelism is provided by independent service workers, not CPU thread speedup.

For LLM execution install `--extra llm`, configure `EVALMUT_MODEL` and provider
credentials on the worker, and use `pointwise` or `pairwise`. The Compose example
starts with deterministic execution; pass your selected provider environment to
the worker when configuring a model. Paid model runs are operator controlled.

## Verification

```bash
make install
make verify
EVALMUT_TEST_DATABASE_URL='postgresql+asyncpg://user:password@localhost:5432/testdb' \
  uv run pytest tests/test_service.py
```

The PostgreSQL test database must be isolated from real results. Tests cover
concurrent idempotent submissions, competing claims, ownership fencing, expired
leases, authentication, payload validation, changed datasets, and challenge
execution. Async tests cover cancellation, timeouts, ordering, bounded parallelism,
redacted errors, and refusal to score failed predictions.

## Operational scope and next gates

This release provides a single-tenant service and an initial database schema.
Before an internet-facing deployment, add versioned Alembic migrations, per-user
authorization and quotas, TLS termination, request rate limits, database backups,
retention, metrics/tracing, and measured load/chaos tests. Keep clocks synchronized
across workers because lease timestamps currently use host clocks. Dataset files
must be immutable during deployment. A privileged operator controls model access.

There is no exactly-once guarantee for external model requests. Explicit retry
requires a new idempotency key and may repeat provider cost. Database results are
fenced by claim token; external requests can outlive a cancelled connection.

Native Ragas/DeepEval integrations, human-validity evidence, retrieval ablations,
and measured latency under load remain independently tracked benchmark work.

## Design references

- [FastAPI async execution](https://fastapi.tiangolo.com/async/)
- [SQLAlchemy async transactions](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Python TaskGroup and timeouts](https://docs.python.org/3.12/library/asyncio-task.html)
- [LiteLLM asynchronous completion](https://docs.litellm.ai/docs/completion/stream)
