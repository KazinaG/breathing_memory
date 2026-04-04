from __future__ import annotations

from typing import Any

from ..core.types import (
    FeedbackResult,
    FragmentView,
    ReadActiveCollaborationPolicyResponse,
    SearchItemView,
    SearchResponse,
    StatsResult,
)


def _normalize_payload_mode(payload_mode: str) -> str:
    if payload_mode not in {"slim", "debug"}:
        raise ValueError(f"Unsupported payload mode: {payload_mode}")
    return payload_mode


def fragment_view_to_payload(view: FragmentView, *, payload_mode: str = "debug") -> dict[str, Any]:
    mode = _normalize_payload_mode(payload_mode)
    if mode == "slim":
        return {
            "id": view.id,
            "anchor_id": view.anchor_id,
            "reply_to": view.reply_to,
            "kind": view.kind,
        }
    return {
        "id": view.id,
        "anchor_id": view.anchor_id,
        "reply_to": view.reply_to,
        "kind": view.kind,
        "content": view.content,
        "content_length": view.content_length,
        "layer": view.layer,
        "compression_fail_count": view.compression_fail_count,
        "reference_score": view.reference_score,
        "confidence_score": view.confidence_score,
        "search_priority": view.search_priority,
    }


def search_item_view_to_payload(view: SearchItemView, *, payload_mode: str = "debug") -> dict[str, Any]:
    mode = _normalize_payload_mode(payload_mode)
    payload = {
        "id": view.id,
        "anchor_id": view.anchor_id,
        "parent_id": view.parent_id,
        "actor": view.actor,
        "reply_to": view.reply_to,
        "kind": view.kind,
        "content": view.content,
        "layer": view.layer,
    }
    if mode == "debug":
        payload["content_length"] = view.content_length
        payload["reference_score"] = view.reference_score
        payload["confidence_score"] = view.confidence_score
        payload["search_priority"] = view.search_priority
    if view.diagnostics is not None:
        payload["diagnostics"] = view.diagnostics
    return payload


def search_response_to_payload(response: SearchResponse, *, payload_mode: str = "debug") -> dict[str, Any]:
    payload = {
        "items": [search_item_view_to_payload(item, payload_mode=payload_mode) for item in response.items],
        "count": response.count,
    }
    if response.status is not None:
        payload["status"] = response.status
    return payload


def read_active_collaboration_policy_response_to_payload(
    response: ReadActiveCollaborationPolicyResponse,
    *,
    payload_mode: str = "debug",
) -> dict[str, Any]:
    return {
        "items": [search_item_view_to_payload(item, payload_mode=payload_mode) for item in response.items],
        "count": response.count,
        "token_budget": response.token_budget,
        "used_token_budget": response.used_token_budget,
        "truncated": response.truncated,
    }


def feedback_result_to_payload(result: FeedbackResult, *, payload_mode: str = "debug") -> dict[str, Any]:
    mode = _normalize_payload_mode(payload_mode)
    if mode == "slim":
        return {
            "fragment_id": result.fragment_id,
            "verdict": result.verdict,
        }
    return {
        "fragment_id": result.fragment_id,
        "verdict": result.verdict,
        "confidence_score": result.confidence_score,
        "search_priority": result.search_priority,
    }


def stats_result_to_payload(result: StatsResult) -> dict[str, Any]:
    return {
        "fragment_count": result.fragment_count,
        "working_count": result.working_count,
        "holding_count": result.holding_count,
        "working_usage": result.working_usage,
        "holding_usage": result.holding_usage,
        "working_budget": result.working_budget,
        "holding_budget": result.holding_budget,
        "working_ratio": result.working_ratio,
        "recent_compress_count": result.recent_compress_count,
        "recent_delete_count": result.recent_delete_count,
        "parameters": result.parameters,
    }


def result_to_payload(result: Any, *, payload_mode: str = "debug") -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, FragmentView):
        return fragment_view_to_payload(result, payload_mode=payload_mode)
    if isinstance(result, SearchResponse):
        return search_response_to_payload(result, payload_mode=payload_mode)
    if isinstance(result, ReadActiveCollaborationPolicyResponse):
        return read_active_collaboration_policy_response_to_payload(result, payload_mode=payload_mode)
    if isinstance(result, FeedbackResult):
        return feedback_result_to_payload(result, payload_mode=payload_mode)
    if isinstance(result, StatsResult):
        return stats_result_to_payload(result)
    raise TypeError(f"Unsupported payload result type: {type(result)!r}")
