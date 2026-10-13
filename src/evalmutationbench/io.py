from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from .models import EvaluatorPrediction, TracePair


def read_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json(exclude_none=True) + "\n")


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pairs(path: Path) -> list[TracePair]:
    return read_jsonl(path, TracePair)


def load_predictions(path: Path) -> list[EvaluatorPrediction]:
    return read_jsonl(path, EvaluatorPrediction)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
