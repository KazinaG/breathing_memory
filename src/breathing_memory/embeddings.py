from __future__ import annotations

from array import array
import math
from typing import Optional, Protocol, Sequence


DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingBackend(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbeddingBackend:
    def __init__(self, model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
        )
        return [list(map(float, vector)) for vector in vectors]


class StubEmbeddingBackend:
    def __init__(self, vectors_by_text: dict[str, Sequence[float]]):
        self.vectors_by_text = {key: list(map(float, value)) for key, value in vectors_by_text.items()}

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = self.vectors_by_text.get(text)
            if vector is None:
                raise KeyError(f"missing stub embedding for text: {text}")
            vectors.append(list(vector))
        return vectors


def try_create_default_embedding_backend() -> Optional[EmbeddingBackend]:
    try:
        return SentenceTransformerEmbeddingBackend()
    except Exception:
        return None


def pack_embedding(vector: Sequence[float]) -> bytes:
    values = array("f", (float(component) for component in vector))
    return values.tobytes()


def unpack_embedding(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimension mismatch")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
