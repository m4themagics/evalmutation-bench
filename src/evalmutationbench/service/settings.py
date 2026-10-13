from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    database_url: SecretStr
    api_token: SecretStr = Field(min_length=32)
    data_dir: Path = Path("data")
    model: str | None = None
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
        )
