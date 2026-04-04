from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping

from .runtime import resolve_db_path

TOTAL_CAPACITY_MB_ENV_VAR = "BREATHING_MEMORY_TOTAL_CAPACITY_MB"
ACP_TOKEN_BUDGET_ENV_VAR = "BREATHING_MEMORY_ACP_TOKEN_BUDGET"
MCP_PAYLOAD_MODE_ENV_VAR = "BREATHING_MEMORY_MCP_PAYLOAD_MODE"


def resolve_total_capacity_mb(env: Mapping[str, str] | None = None) -> float:
    environment = env if env is not None else os.environ
    raw = environment.get(TOTAL_CAPACITY_MB_ENV_VAR, "").strip()
    if not raw:
        return 32
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{TOTAL_CAPACITY_MB_ENV_VAR} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{TOTAL_CAPACITY_MB_ENV_VAR} must be a positive number")
    return value


def resolve_default_acp_token_budget(env: Mapping[str, str] | None = None) -> int:
    environment = env if env is not None else os.environ
    raw = environment.get(ACP_TOKEN_BUDGET_ENV_VAR, "").strip()
    if not raw:
        return 512
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{ACP_TOKEN_BUDGET_ENV_VAR} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{ACP_TOKEN_BUDGET_ENV_VAR} must be a positive integer")
    return value


def resolve_mcp_payload_mode(env: Mapping[str, str] | None = None) -> str:
    environment = env if env is not None else os.environ
    raw = environment.get(MCP_PAYLOAD_MODE_ENV_VAR, "").strip()
    if not raw:
        return "slim"
    if raw not in {"slim", "debug"}:
        raise ValueError(f"{MCP_PAYLOAD_MODE_ENV_VAR} must be one of: slim, debug")
    return raw


@dataclass(frozen=True)
class MemoryConfig:
    db_path: Path = field(default_factory=resolve_db_path)
    total_capacity_mb: float = field(default_factory=resolve_total_capacity_mb)
    default_acp_token_budget: int = field(default_factory=resolve_default_acp_token_budget)
    mcp_payload_mode: str = field(default_factory=resolve_mcp_payload_mode)
    retrieval_mode: str = "auto"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class EngineTuning:
    failure_penalty_base: float = 2.0
    initial_working_ratio: float = 0.5
    working_ratio_floor: float = 0.3
    rate_window_size: int = 64
    compression_ratio: float = 0.8
    epsilon: float = 1e-9
    default_result_count: int = 4
    default_search_effort: int = 32
    ann_wait_timeout_ms: int = 500
