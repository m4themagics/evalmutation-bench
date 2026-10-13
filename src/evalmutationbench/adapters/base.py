from __future__ import annotations

import asyncio
import importlib
from abc import ABC, abstractmethod
from typing import Any

from ..models import EvaluatorPrediction, TracePair


class EvaluatorAdapter(ABC):
    name = "abstract"
    version = "0"

    @abstractmethod
    def evaluate(self, pair: TracePair) -> EvaluatorPrediction:
        """Evaluate one control/candidate trace pair."""

    async def aevaluate(self, pair: TracePair) -> EvaluatorPrediction:
        """Compatibility path; sync plugins must be thread safe and bound their own I/O.

        Cancelling a thread await cannot terminate underlying synchronous work.
        Network adapters should override this with cancellable native async I/O.
        """
        return await asyncio.to_thread(self.evaluate, pair)


def load_adapter(spec: str, config: dict[str, Any] | None = None) -> EvaluatorAdapter:
    config = config or {}
    if spec == "deterministic":
        from .deterministic import DeterministicAdapter

        return DeterministicAdapter(**config)
    if spec == "pointwise":
        from .llm import LLMJudgeAdapter

        return LLMJudgeAdapter(mode="pointwise", **config)
    if spec == "pairwise":
        from .llm import LLMJudgeAdapter

        return LLMJudgeAdapter(mode="pairwise", **config)
    if spec == "ragas":
        from .frameworks import FrameworkJudgeAdapter

        return FrameworkJudgeAdapter(framework="ragas", **config)
    if spec == "deepeval":
        from .frameworks import FrameworkJudgeAdapter

        return FrameworkJudgeAdapter(framework="deepeval", **config)
    if ":" not in spec:
        raise ValueError("adapter must be a built-in name or module:factory")
    module_name, factory_name = spec.split(":", maxsplit=1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    adapter = factory(config)
    if not isinstance(adapter, EvaluatorAdapter):
        raise TypeError(f"{spec} did not create an EvaluatorAdapter")
    return adapter
