from __future__ import annotations

import unittest
from unittest.mock import patch
import threading

from breathing_memory.embeddings import SentenceTransformerEmbeddingBackend, try_create_default_embedding_backend


class _FakeSentenceTransformer:
    init_calls: list[str] = []
    encode_calls: list[list[str]] = []

    def __init__(self, model_name: str):
        self.init_calls.append(model_name)

    def encode(self, texts, normalize_embeddings=True):
        self.encode_calls.append(list(texts))
        return [[float(index), float(index + 1)] for index, _ in enumerate(texts, start=1)]


class _FakeSentenceTransformersModule:
    SentenceTransformer = _FakeSentenceTransformer


class EmbeddingsTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSentenceTransformer.init_calls = []
        _FakeSentenceTransformer.encode_calls = []

    def test_sentence_transformer_backend_loads_model_on_first_embed(self) -> None:
        with patch(
            "breathing_memory.embeddings.importlib.import_module",
            return_value=_FakeSentenceTransformersModule(),
        ) as import_module:
            backend = SentenceTransformerEmbeddingBackend("test-model")

            self.assertEqual(_FakeSentenceTransformer.init_calls, [])

            vectors = backend.embed_texts(["alpha", "beta"])

            self.assertEqual(vectors, [[1.0, 2.0], [2.0, 3.0]])
            self.assertEqual(_FakeSentenceTransformer.init_calls, ["test-model"])
            self.assertEqual(_FakeSentenceTransformer.encode_calls, [["alpha", "beta"]])
            import_module.assert_called_once_with("sentence_transformers")

    def test_sentence_transformer_backend_reuses_loaded_model(self) -> None:
        with patch(
            "breathing_memory.embeddings.importlib.import_module",
            return_value=_FakeSentenceTransformersModule(),
        ) as import_module:
            backend = SentenceTransformerEmbeddingBackend("test-model")

            backend.embed_texts(["alpha"])
            backend.embed_texts(["beta"])

            self.assertEqual(_FakeSentenceTransformer.init_calls, ["test-model"])
            self.assertEqual(_FakeSentenceTransformer.encode_calls, [["alpha"], ["beta"]])
            import_module.assert_called_once_with("sentence_transformers")

    def test_try_create_default_embedding_backend_returns_none_without_dependency(self) -> None:
        with patch("breathing_memory.embeddings.importlib.util.find_spec", return_value=None):
            backend = try_create_default_embedding_backend()

        self.assertIsNone(backend)

    def test_try_create_default_embedding_backend_does_not_load_model(self) -> None:
        with patch("breathing_memory.embeddings.importlib.util.find_spec", return_value=object()), patch(
            "breathing_memory.embeddings.importlib.import_module"
        ) as import_module:
            backend = try_create_default_embedding_backend()

        self.assertIsInstance(backend, SentenceTransformerEmbeddingBackend)
        import_module.assert_not_called()

    def test_background_warmup_and_foreground_embed_share_one_initialization(self) -> None:
        init_started = threading.Event()
        finish_init = threading.Event()

        class BlockingSentenceTransformer:
            init_calls: list[str] = []
            encode_calls: list[list[str]] = []

            def __init__(self, model_name: str):
                self.init_calls.append(model_name)
                init_started.set()
                finish_init.wait(timeout=5)

            def encode(self, texts, normalize_embeddings=True):
                del normalize_embeddings
                self.encode_calls.append(list(texts))
                return [[float(index)] for index, _ in enumerate(texts, start=1)]

        class FakeModule:
            SentenceTransformer = BlockingSentenceTransformer

        with patch(
            "breathing_memory.embeddings.importlib.import_module",
            return_value=FakeModule(),
        ):
            backend = SentenceTransformerEmbeddingBackend("test-model")
            self.assertTrue(backend.start_background_warmup())
            self.assertTrue(init_started.wait(timeout=5))

            vectors_holder: dict[str, list[list[float]]] = {}

            def _embed() -> None:
                vectors_holder["vectors"] = backend.embed_texts(["alpha"])

            foreground = threading.Thread(target=_embed)
            foreground.start()
            finish_init.set()
            foreground.join(timeout=5)
            self.assertFalse(foreground.is_alive())

        self.assertEqual(BlockingSentenceTransformer.init_calls, ["test-model"])
        self.assertEqual(vectors_holder["vectors"], [[1.0]])
        self.assertEqual(BlockingSentenceTransformer.encode_calls, [["alpha"]])

    def test_background_warmup_failure_does_not_block_later_retry(self) -> None:
        class FakeSentenceTransformer:
            init_calls: list[str] = []

            def __init__(self, model_name: str):
                self.init_calls.append(model_name)

            def encode(self, texts, normalize_embeddings=True):
                del normalize_embeddings
                return [[float(index)] for index, _ in enumerate(texts, start=1)]

        class FakeModule:
            SentenceTransformer = FakeSentenceTransformer

        import_calls = [ImportError("boom"), FakeModule()]
        errors: list[Exception] = []
        backend = SentenceTransformerEmbeddingBackend("test-model")

        with patch(
            "breathing_memory.embeddings.importlib.import_module",
            side_effect=import_calls,
        ):
            self.assertTrue(backend.start_background_warmup(on_error=errors.append))
            for _ in range(50):
                if errors:
                    break
                threading.Event().wait(0.01)
            vectors = backend.embed_texts(["alpha"])

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ImportError)
        self.assertEqual(FakeSentenceTransformer.init_calls, ["test-model"])
        self.assertEqual(vectors, [[1.0]])


if __name__ == "__main__":
    unittest.main()
