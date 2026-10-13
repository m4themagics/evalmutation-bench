import asyncio
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict

from ..io import dataset_sha256
from ..retail.models import ApprovalRequest, RetailRequest
from ..retail.store import RetailConflict, RetailStore
from .settings import Settings
from .store import IdempotencyConflict, RunStore


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    split: Literal["development", "validation", "challenge"] = "development"
    adapter: Literal["deterministic", "pointwise", "pairwise"] = "deterministic"


class RunResponse(BaseModel):
    run_id: UUID
    status: Literal["queued", "running", "completed", "failed"]
    request: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = RunStore(settings.database_url.get_secret_value())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await store.close()

    app = FastAPI(title="EvalMutationBench API", version="0.1.0", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    async def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if credentials is None or not secrets.compare_digest(
            credentials.credentials, settings.api_token.get_secret_value()
        ):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, str]:
        try:
            async with asyncio.timeout(2):
                await store.healthy()
        except Exception:
            raise HTTPException(503, "Database unavailable") from None
        return {"status": "ok"}

    @app.post(
        "/v1/runs",
        status_code=202,
        response_model=RunResponse,
        dependencies=[Depends(authenticate)],
    )
    async def submit(
        request: RunRequest,
        idempotency_key: Annotated[str, Header(min_length=1, max_length=128)],
    ) -> dict[str, Any]:
        if request.adapter != "deterministic" and not settings.model:
            raise HTTPException(422, "Operator has not configured an LLM model")
        dataset = settings.data_dir / f"{request.split}.jsonl"
        try:
            checksum = await asyncio.to_thread(dataset_sha256, dataset)
        except OSError:
            raise HTTPException(503, "Dataset unavailable") from None
        payload = {
            **request.model_dump(),
            "dataset_sha256": checksum,
            "model": settings.model if request.adapter != "deterministic" else None,
        }
        try:
            return await store.create(idempotency_key, payload)
        except IdempotencyConflict:
            raise HTTPException(409, "Idempotency key belongs to a different request") from None

    @app.get("/v1/runs/{run_id}", response_model=RunResponse, dependencies=[Depends(authenticate)])
    async def get_run(run_id: UUID) -> dict[str, Any]:
        row = await store.get(str(run_id))
        if row is None:
            raise HTTPException(404, "Run not found")
        return row

    if settings.retail_enabled:
        retail = RetailStore(store)

        @app.post(
            "/v1/retail/sessions",
            status_code=202,
            response_model=RunResponse,
            dependencies=[Depends(authenticate)],
        )
        async def start_retail(
            request: RetailRequest,
            idempotency_key: Annotated[str, Header(min_length=1, max_length=128)],
        ) -> dict[str, Any]:
            try:
                return await retail.start(idempotency_key, request, {
                    "mode": settings.retail_retrieval_mode,
                    "embedding_model": settings.retail_embedding_model,
                    "reranker_model": settings.retail_reranker_model,
                })
            except IdempotencyConflict:
                raise HTTPException(409, "Idempotency key belongs to a different request") from None

        @app.get("/v1/retail/sessions/{session_id}", dependencies=[Depends(authenticate)])
        async def get_retail(session_id: UUID) -> dict[str, Any]:
            session = await retail.get_session(str(session_id))
            if session is None:
                raise HTTPException(404, "Session not found")
            return session

        @app.post(
            "/v1/retail/sessions/{session_id}/approval",
            status_code=202,
            response_model=RunResponse,
            dependencies=[Depends(authenticate)],
        )
        async def approve_retail(
            session_id: UUID,
            request: ApprovalRequest,
            idempotency_key: Annotated[str, Header(min_length=1, max_length=128)],
        ) -> dict[str, Any]:
            try:
                return await retail.approve(idempotency_key, str(session_id), request)
            except (RetailConflict, IdempotencyConflict):
                raise HTTPException(409, "Approval conflicts with the current proposal") from None

    return app
