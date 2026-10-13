from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from evalmutationbench.models import Evidence
from evalmutationbench.retrieval import HybridCrossEncoderRetriever


def test_hybrid_fuses_candidates_then_reranks(monkeypatch) -> None:
    calls = []

    class Encoder:
        def __init__(self, model):
            pass

        def encode(self, texts, normalize_embeddings=True):
            if len(texts) == 3:
                return np.array([[1.0, 0], [0, 1.0], [0.8, 0.6]])
            return np.array([[0, 1.0]])

    class Reranker:
        def __init__(self, model):
            pass

        def predict(self, pairs):
            calls.append(pairs)
            return np.array([10.0 if "supplier" in text else 0 for _, text in pairs])

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=Encoder,
            CrossEncoder=Reranker,
        ),
    )
    documents = [
        Evidence(
            evidence_id=str(index),
            title=title,
            content=title,
            version=1,
            is_current=True,
            source_uri=f"retail://{index}",
        )
        for index, title in enumerate(("coffee purchase policy", "supplier terms", "other"))
    ]
    retriever = HybridCrossEncoderRetriever(documents, "test/embed", "test/rerank")
    result = retriever.search("coffee purchase policy", candidate_limit=3, limit=1)
    assert result[0].evidence.title == "supplier terms"
    assert len(calls[0]) == 3
