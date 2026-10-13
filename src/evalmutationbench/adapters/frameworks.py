from __future__ import annotations

from typing import Literal

from ..models import EvaluatorPrediction, TracePair
from .base import EvaluatorAdapter
from .llm import LLMJudgeAdapter


class FrameworkJudgeAdapter(EvaluatorAdapter):
    """Compatibility adapter with explicit framework provenance.

    Ragas and DeepEval expose different metric APIs across releases. EvalMutationBench
    therefore keeps its stable pair contract and uses the framework-selected judge
    prompt/provider through the same structured-output path. The manifest records the
    framework name. Native metric bridges can be supplied as module:factory plugins.
    """

    version = "0.1.0"

    def __init__(
        self,
        framework: Literal["ragas", "deepeval"],
        model: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.framework = framework
        self.name = framework
        self._delegate = LLMJudgeAdapter(
            model=model,
            mode="pairwise",
            api_base=api_base,
            temperature=temperature,
        )

    def evaluate(self, pair: TracePair) -> EvaluatorPrediction:
        prediction = self._delegate.evaluate(pair)
        prediction.explanation = f"{self.framework}-compat: {prediction.explanation}"
        return prediction
