from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    database_url: SecretStr
    api_token: SecretStr = Field(min_length=32)
    data_dir: Path = Path("data")
    model: str | None = None
    retail_enabled: bool = False
    retail_retrieval_mode: Literal["lexical", "hybrid"] = "lexical"
    retail_embedding_model: str | None = None
    retail_reranker_model: str | None = None

    @model_validator(mode="after")
    def check_retrieval_models(self) -> Settings:
        if self.retail_retrieval_mode == "hybrid" and (
            not self.retail_embedding_model or not self.retail_reranker_model
        ):
            raise ValueError("hybrid mode requires pinned embedding and reranker model paths")
        return self
    concurrency: int = Field(default=4, ge=1, le=32)
    pair_timeout_seconds: float = Field(default=60, gt=0, le=600)
    run_timeout_seconds: float = Field(default=600, gt=0, le=3600)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=SecretStr(os.environ["EVALMUT_DATABASE_URL"]),
            api_token=SecretStr(os.environ["EVALMUT_SERVICE_TOKEN"]),
            data_dir=Path(os.getenv("EVALMUT_DATA_DIR", "data")),
            model=os.getenv("EVALMUT_MODEL"),
            retail_enabled=os.getenv("EVALMUT_RETAIL_ENABLED", "0") == "1",
            retail_retrieval_mode=cast(
                Literal["lexical", "hybrid"], os.getenv("EVALMUT_RETAIL_RETRIEVAL_MODE", "lexical")
            ),
            retail_embedding_model=os.getenv("EVALMUT_RETAIL_EMBEDDING_MODEL"),
            retail_reranker_model=os.getenv("EVALMUT_RETAIL_RERANKER_MODEL"),
        )
