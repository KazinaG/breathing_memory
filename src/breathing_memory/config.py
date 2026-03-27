from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .runtime import resolve_db_path


@dataclass(frozen=True)
class MemoryConfig:
    db_path: Path = field(default_factory=resolve_db_path)
    total_capacity_mb: float = 32
    retrieval_mode: str = "auto"


@dataclass(frozen=True)
class EngineTuning:
    failure_penalty_base: float = 2.0
    initial_working_ratio: float = 0.5
    working_ratio_floor: float = 0.3
    rate_window_size: int = 64
    compression_ratio: float = 0.8
    epsilon: float = 1e-9
    default_result_count: int = 8
    default_search_effort: int = 32
