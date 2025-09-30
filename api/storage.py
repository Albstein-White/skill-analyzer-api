"""Utility helpers for persisting reports and session metadata.

The production deployment should ideally swap this module for a proper
database-backed implementation.  For now we use simple JSON files stored on
disk to keep the API stateless across restarts and to support shareable
report links.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_ROOT = Path(os.getenv("DATA_DIR", "data")).resolve()
REPORTS_DIR = DATA_ROOT / "reports"
REPORT_INDEX_PATH = DATA_ROOT / "reports_index.json"
ACTIVE_SESSIONS_PATH = DATA_ROOT / "sessions_active.json"

_LOCK = threading.Lock()


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_report(report_id: str, report: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Persist the rendered report JSON and its index metadata."""

    _ensure_dirs()
    report_path = REPORTS_DIR / f"{report_id}.json"

    with _LOCK:
        index: Dict[str, Dict[str, Any]] = _read_json(REPORT_INDEX_PATH, {})
        index[report_id] = metadata
        _write_json(REPORT_INDEX_PATH, index)

    _write_json(report_path, report)


def load_report(report_id: str) -> Optional[Dict[str, Any]]:
    path = REPORTS_DIR / f"{report_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def delete_report(report_id: str) -> bool:
    removed = False
    with _LOCK:
        index: Dict[str, Dict[str, Any]] = _read_json(REPORT_INDEX_PATH, {})
        if report_id in index:
            index.pop(report_id, None)
            _write_json(REPORT_INDEX_PATH, index)
            removed = True
    report_path = REPORTS_DIR / f"{report_id}.json"
    if report_path.exists():
        try:
            report_path.unlink()
        except Exception:
            pass
    return removed


def list_reports_for_user(user_id: str) -> List[Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = _read_json(REPORT_INDEX_PATH, {})
    out: List[Dict[str, Any]] = []
    for rid, meta in index.items():
        if meta.get("userId") == user_id:
            item = {"id": rid}
            item.update({k: v for k, v in meta.items() if k != "id"})
            out.append(item)
    out.sort(key=lambda r: r.get("createdAt", ""), reverse=True)
    return out


def find_report_by_session(session_id: str) -> Optional[Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = _read_json(REPORT_INDEX_PATH, {})
    for rid, meta in index.items():
        if meta.get("sessionId") == session_id:
            report = load_report(rid)
            if report:
                return report
    return None


def _load_sessions() -> Dict[str, Dict[str, Any]]:
    return _read_json(ACTIVE_SESSIONS_PATH, {})


def record_active_session(session_id: str, payload: Dict[str, Any]) -> None:
    if not payload.get("userId"):
        return
    with _LOCK:
        sessions = _load_sessions()
        sessions[session_id] = payload
        _write_json(ACTIVE_SESSIONS_PATH, sessions)


def update_active_session(session_id: str, updates: Dict[str, Any]) -> None:
    with _LOCK:
        sessions = _load_sessions()
        if session_id not in sessions:
            return
        sessions[session_id].update(updates)
        _write_json(ACTIVE_SESSIONS_PATH, sessions)


def clear_active_session(session_id: str) -> None:
    with _LOCK:
        sessions = _load_sessions()
        if session_id in sessions:
            sessions.pop(session_id, None)
            _write_json(ACTIVE_SESSIONS_PATH, sessions)


def active_sessions_for_user(user_id: str) -> List[Dict[str, Any]]:
    sessions = _load_sessions()
    out: List[Dict[str, Any]] = []
    for payload in sessions.values():
        if payload.get("userId") == user_id:
            out.append(payload)
    out.sort(key=lambda r: r.get("startedAt", ""), reverse=True)
    return out


def load_all_active_sessions() -> Dict[str, Dict[str, Any]]:
    return _load_sessions()

