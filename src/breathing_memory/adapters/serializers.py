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


def fragment_view_to_payload(view: FragmentView) -> dict[str, Any]:
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


def search_item_view_to_payload(view: SearchItemView) -> dict[str, Any]:
    payload = {
        "id": view.id,
        "anchor_id": view.anchor_id,
        "parent_id": view.parent_id,
        "actor": view.actor,
        "reply_to": view.reply_to,
        "kind": view.kind,
        "content": view.content,
        "content_length": view.content_length,
        "layer": view.layer,
        "reference_score": view.reference_score,
        "confidence_score": view.confidence_score,
        "search_priority": view.search_priority,
    }
    if view.diagnostics is not None:
        payload["diagnostics"] = view.diagnostics
    return payload


def search_response_to_payload(response: SearchResponse) -> dict[str, Any]:
    payload = {
        "items": [search_item_view_to_payload(item) for item in response.items],
        "count": response.count,
    }
    if response.status is not None:
        payload["status"] = response.status
    return payload


def read_active_collaboration_policy_response_to_payload(
    response: ReadActiveCollaborationPolicyResponse,
) -> dict[str, Any]:
    return {
        "items": [search_item_view_to_payload(item) for item in response.items],
        "count": response.count,
        "token_budget": response.token_budget,
        "used_token_budget": response.used_token_budget,
        "truncated": response.truncated,
    }


def feedback_result_to_payload(result: FeedbackResult) -> dict[str, Any]:
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


def result_to_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, FragmentView):
        return fragment_view_to_payload(result)
    if isinstance(result, SearchResponse):
        return search_response_to_payload(result)
    if isinstance(result, ReadActiveCollaborationPolicyResponse):
        return read_active_collaboration_policy_response_to_payload(result)
    if isinstance(result, FeedbackResult):
        return feedback_result_to_payload(result)
    if isinstance(result, StatsResult):
        return stats_result_to_payload(result)
    raise TypeError(f"Unsupported payload result type: {type(result)!r}")
