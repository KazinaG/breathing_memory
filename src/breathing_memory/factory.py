from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional

from .config import (
    EngineTuning,
    MemoryConfig,
    resolve_default_acp_token_budget,
    resolve_mcp_payload_mode,
    resolve_total_capacity_mb,
)
from .core.ports import AnnIndex, CompressionBackend, EmbeddingBackend, Store
from .core.service import BreathingMemoryEngine as CoreBreathingMemoryEngine
from .runtime import resolve_db_path


def resolve_memory_config(
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    retrieval_mode: str | None = None,
    embedding_model: str | None = None,
) -> MemoryConfig:
    config = MemoryConfig(
        db_path=resolve_db_path(cwd=cwd, env=env),
        total_capacity_mb=resolve_total_capacity_mb(env=env),
        default_acp_token_budget=resolve_default_acp_token_budget(env=env),
        mcp_payload_mode=resolve_mcp_payload_mode(env=env),
    )
    if retrieval_mode is None and embedding_model is None:
        return config
    return replace(
        config,
        retrieval_mode=config.retrieval_mode if retrieval_mode is None else retrieval_mode,
        embedding_model=config.embedding_model if embedding_model is None else embedding_model,
    )


def create_core_engine(
    *,
    config: Optional[MemoryConfig] = None,
    tuning: Optional[EngineTuning] = None,
    store: Optional[Store] = None,
    compression_backend: Optional[CompressionBackend] = None,
    embedding_backend: Optional[EmbeddingBackend] = None,
    ann_index: Optional[AnnIndex] = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    retrieval_mode: str | None = None,
    embedding_model: str | None = None,
) -> CoreBreathingMemoryEngine:
    effective_config = config
    if effective_config is None:
        effective_config = resolve_memory_config(
            cwd=cwd,
            env=env,
            retrieval_mode=retrieval_mode,
            embedding_model=embedding_model,
        )
    return CoreBreathingMemoryEngine(
        config=effective_config,
        tuning=tuning,
        store=store,
        compression_backend=compression_backend,
        embedding_backend=embedding_backend,
        ann_index=ann_index,
    )


__all__ = [
    "create_core_engine",
    "resolve_memory_config",
]
