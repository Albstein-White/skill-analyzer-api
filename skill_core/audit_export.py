"""Helpers to export per-step audit traces in JSON/CSV formats."""
from __future__ import annotations

from typing import Iterable, List, Dict, Any
import csv
import io

_FIELDS: tuple[str, ...] = (
    "t",
    "domain",
    "item_id",
    "type",
    "b",
    "level_before",
    "level_after",
    "correct_or_r",
    "theta_before",
    "theta_after",
    "se_after",
    "latency_ms",
)


def _normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in _FIELDS:
        val = event.get(key)
        if key in {"b", "level_before", "level_after", "latency_ms"}:
            try:
                out[key] = int(val)
            except (TypeError, ValueError):
                out[key] = 0
        elif key in {"correct_or_r", "theta_before", "theta_after", "se_after"}:
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                out[key] = 0.0
        else:
            out[key] = "" if val is None else str(val)
    return out


def to_json(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a JSON-safe payload for audit export."""

    normalized: List[Dict[str, Any]] = [_normalize_event(evt or {}) for evt in events]
    return {"events": normalized}


def to_csv(events: Iterable[Dict[str, Any]]) -> str:
    """Render audit events as CSV with a fixed header."""

    normalized = [_normalize_event(evt or {}) for evt in events]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_FIELDS)
    writer.writeheader()
    for row in normalized:
        writer.writerow(row)
    return buf.getvalue()


__all__ = ["to_json", "to_csv"]
