# Roadmap and release evidence

## v0.1 — implemented foundation

- contracts, mutation engine, 300-pair generator;
- public dev/validation and label-free challenge splits;
- deterministic and structured LLM adapter paths;
- statistics, reports, schemas, tests, CI, annotation workflow.

## Service engineering — implemented and separately verified

- typed FastAPI submission and status API with bearer authentication;
- durable PostgreSQL/SQLAlchemy queue and independent async worker;
- idempotent requests, atomic claims, expiring leases, and stale-owner fencing;
- bounded concurrency, pair/run deadlines, cancellation, and sanitized errors;
- refusal to score failed/missing/duplicate predictions;
- Docker Compose, non-root image, and PostgreSQL integration CI.

See [service guide](docs/service.md) for commands and operational boundaries.

## Service engineering — remaining acceptance work

- Alembic migrations and upgrade/rollback tests before evolving the initial schema;
- authenticated tenants, quotas, TLS, and API rate limits before public service hosting;
- OpenTelemetry, backup/restore, result retention, and load/chaos evidence;
- Helm/Kubernetes deployment only with deployment tests and an operating use case;
- broker integration only when delivery/routing requirements justify a broker;
- CPU multiprocessing only for measured CPU bottlenecks.

This project is public code, not claimed production banking experience. Its
service demonstrates Python backend design; benchmark validity still requires
the reviewer and external model work below.

## Retail reference — first milestone implemented

- typed SKU/horizon requests and database-backed inventory/source tools;
- proposals with calculation, citations, and immutable source/proposal hashes;
- durable LangGraph PostgreSQL checkpoints and human interrupt/resume;
- transactional approval enqueue and idempotent purchase tool;
- inventory/source freshness checks and separate session state;
- real worker restart acceptance and reference-trace export to mutation harness;
- optional hybrid retrieval with fusion/reranking, requiring explicit model configuration.

The first milestone is an offline deterministic reference on synthetic retail
fixtures. See [retail.md](docs/retail.md) for evidence and limits. Provider-backed
planning, streaming/retries/budgets, measured retrieval quality, and load tests
remain subsequent milestones. No production retail or paid model result is claimed.

## v0.5 — human validation and model experiments

- three independent reviewers for every pair (900 judgments);
- adjudication of all disagreements;
- Krippendorff's alpha and Fleiss' kappa report;
- native version-pinned Ragas and DeepEval plugins;
- baseline matrix containing at least one API and one local open-weight evaluator;
- bias ablations for order, verbosity, and self-preference.

## v1.0 — public challenge release

- maintainer-held human-adjudicated challenge labels;
- signed leaderboard artifacts and reproducibility bundle;
- external reproduction of one local and one BYOK run;
- paper-style technical report and archived release assets.

No milestone is marked complete from generated labels or planned experiments alone.

