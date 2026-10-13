from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models import Evidence

TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass(frozen=True)
class RankedEvidence:
    evidence: Evidence
    score: float


class BM25Retriever:
    def __init__(self, documents: list[Evidence], k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [_tokens(f"{item.title} {item.content}") for item in documents]
        self.average_length = sum(map(len, self.tokens)) / len(self.tokens)
        self.document_frequency = Counter(token for row in map(set, self.tokens) for token in row)

    def search(self, query: str, limit: int = 10) -> list[RankedEvidence]:
        query_tokens = _tokens(query)
        scored: list[RankedEvidence] = []
        n = len(self.documents)
        for document, tokens in zip(self.documents, self.tokens, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                df = self.document_frequency.get(token, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = frequencies.get(token, 0)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * len(tokens) / self.average_length
                )
                score += idf * (tf * (self.k1 + 1)) / denominator if denominator else 0.0
            scored.append(RankedEvidence(document, score))
        return sorted(scored, key=lambda item: (-item.score, item.evidence.evidence_id))[:limit]


class DenseCrossEncoderRetriever:
    """Optional dense retrieval followed by CrossEncoder reranking."""

    def __init__(
        self,
        documents: list[Evidence],
        embedding_model: str,
        reranker_model: str,
    ) -> None:
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]
                CrossEncoder,
                SentenceTransformer,
            )
        except ImportError as exc:
            raise RuntimeError("install the reference extra: uv sync --extra reference") from exc
        self.documents = documents
        self.encoder: Any = SentenceTransformer(embedding_model)
        self.reranker: Any = CrossEncoder(reranker_model)
        self.texts = [f"{item.title}\n{item.content}" for item in documents]
        self.embeddings: Any = self.encoder.encode(self.texts, normalize_embeddings=True)

    def search(
        self, query: str, limit: int = 10, candidate_limit: int = 50
    ) -> list[RankedEvidence]:
        import numpy as np

        query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.asarray(self.embeddings) @ np.asarray(query_embedding)
        indices = np.argsort(-dense_scores)[:candidate_limit]
        pairs = [(query, self.texts[int(index)]) for index in indices]
        rerank_scores = self.reranker.predict(pairs)
        ranked = sorted(
            (
                RankedEvidence(self.documents[int(index)], float(score))
                for index, score in zip(indices, rerank_scores, strict=True)
            ),
            key=lambda item: -item.score,
        )
        return ranked[:limit]


class HybridCrossEncoderRetriever(DenseCrossEncoderRetriever):
    """BM25 + dense reciprocal-rank fusion, followed by CrossEncoder reranking."""

    def search(
        self, query: str, limit: int = 10, candidate_limit: int = 50
    ) -> list[RankedEvidence]:
        import numpy as np

        lexical = BM25Retriever(self.documents).search(query, limit=candidate_limit)
        vector = self.encoder.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.asarray(self.embeddings) @ np.asarray(vector)
        indices = np.argsort(-dense_scores)[:candidate_limit]
        by_id = {document.evidence_id: index for index, document in enumerate(self.documents)}
        fused: dict[int, float] = {}
        for ranking in (
            [by_id[item.evidence.evidence_id] for item in lexical],
            [int(index) for index in indices],
        ):
            for rank, index in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1 / (60 + rank)
        candidates = sorted(fused, key=lambda index: (-fused[index], index))[:candidate_limit]
        scores = self.reranker.predict([(query, self.texts[index]) for index in candidates])
        ranked = [
            RankedEvidence(self.documents[index], float(score))
            for index, score in zip(candidates, scores, strict=True)
        ]
        return sorted(ranked, key=lambda item: (-item.score, item.evidence.evidence_id))[:limit]
