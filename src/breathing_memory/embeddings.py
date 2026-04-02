from __future__ import annotations

from array import array
import importlib
import importlib.util
import math
import threading
from typing import Any, Callable, Optional, Protocol, Sequence


DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingBackend(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def warmup(self) -> None:
        ...

    def start_background_warmup(self, *, on_error: Callable[[Exception], None] | None = None) -> bool:
        ...


class SentenceTransformerEmbeddingBackend:
    def __init__(self, model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL):
        self.model_name = model_name
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._background_warmup_started = False
        self._background_warmup_thread: threading.Thread | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    module = importlib.import_module("sentence_transformers")
                    self._model = module.SentenceTransformer(self.model_name)
        return self._model

    def warmup(self) -> None:
        self._ensure_model()

    def start_background_warmup(self, *, on_error: Callable[[Exception], None] | None = None) -> bool:
        if self._model is not None:
            return False
        with self._warmup_lock:
            if self._model is not None or self._background_warmup_started:
                return False
            self._background_warmup_started = True

            def _target() -> None:
                try:
                    self.warmup()
                except Exception as exc:
                    if on_error is not None:
                        on_error(exc)
                finally:
                    with self._warmup_lock:
                        self._background_warmup_started = False
                        self._background_warmup_thread = None

            thread = threading.Thread(
                target=_target,
                name="breathing-memory-embedding-warmup",
                daemon=True,
            )
            self._background_warmup_thread = thread
            thread.start()
            return True

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._ensure_model().encode(
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

    def warmup(self) -> None:
        return None

    def start_background_warmup(self, *, on_error: Callable[[Exception], None] | None = None) -> bool:
        del on_error
        return False


def try_create_default_embedding_backend() -> Optional[EmbeddingBackend]:
    try:
        if importlib.util.find_spec("sentence_transformers") is None:
            return None
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
