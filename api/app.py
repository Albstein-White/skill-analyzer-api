from __future__ import annotations
from fastapi import FastAPI, HTTPException, Body, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid, os, json, time, pathlib, typing as t

# ---- Azure autoload from .azure_config.json (only if env is missing) ----
def _load_azure_from_json(path: str = ".azure_config.json") -> None:
    need = ["AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_API_VERSION","AZURE_OPENAI_DEPLOYMENT"]
    if all(os.getenv(k) for k in need):
        return
    p = pathlib.Path(path)
    if not p.exists():
        return
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        os.environ.setdefault("USE_LLM_OPEN", "1")
        os.environ.setdefault("LLM_BACKEND", "azure")
        os.environ.setdefault("AZURE_OPENAI_ENDPOINT",   str(cfg.get("endpoint","")))
        os.environ.setdefault("AZURE_OPENAI_API_KEY",    str(cfg.get("api_key","")))
        os.environ.setdefault("AZURE_OPENAI_API_VERSION",str(cfg.get("api_version","")))
        os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", str(cfg.get("deployment","")))
    except Exception:
        pass

_load_azure_from_json()

# ---- Engine imports ----
from skill_core.engine import AdaptiveSession
from skill_core.types import Answer
from skill_core.plan import generate_plan
from skill_core.config import load_config, AUDIT_EXPORT_ENABLED
from skill_core.audit_export import to_json as audit_to_json, to_csv as audit_to_csv
from .storage import (
    active_sessions_for_user,
    clear_active_session,
    delete_report,
    find_report_by_session,
    list_reports_for_user,
    load_all_active_sessions,
    load_report,
    record_active_session,
    save_report,
    update_active_session,
    utcnow_iso,
)

SESS: dict[str, AdaptiveSession] = {}
NEXT_CACHE: dict[str, t.Any] = {}  # sid -> serialized next item
SESSION_INFO: dict[str, dict[str, t.Any]] = {}

for sid, payload in load_all_active_sessions().items():
    SESSION_INFO[sid] = {
        "user_id": payload.get("userId"),
        "run": payload.get("run"),
        "started_at": payload.get("startedAt"),
    }

app = FastAPI(title="Skill Analyzer API")

# api/app.py, after `app = FastAPI(...)`
@app.get("/")
def root():
    return {"status": "ok", "service": "skill-analyzer-api"}

from fastapi.middleware.cors import CORSMiddleware

ALLOWED_ORIGINS = [
    "https://skill-tier.com",
    "https://www.skill-tier.com",
    "http://localhost:3000",
    "https://bolt.new/~/sb1-zfve5vhp",# dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # keep False unless you use cookies
)

# ---- Schemas ----
class StartReq(BaseModel):
    run: str            # "short" | "long"
    llm: str = "none"   # "none" | "azure" | "ollama"
    user_id: str | None = None

class AnswerReq(BaseModel):
    item_id: str
    value: int | str
    rt_ms: int | None = None

class FEAnswer(BaseModel):
    session_id: str
    item_id: str
    answer: int | str
    started_at: float | None = None
    submitted_at: float | None = None

class FEFinish(BaseModel):
    session_id: str

# ---- Helpers ----
def _now_iso() -> str:
    return utcnow_iso()


def _serialize_result(res: t.Any) -> dict[str, t.Any]:
    return json.loads(json.dumps(res, default=lambda o: getattr(o, "__dict__", o)))


def _decorate_report(
    base: dict[str, t.Any],
    *,
    session_id: str,
    user_id: str | None,
    report_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, t.Any]:
    rid = report_id or str(uuid.uuid4())
    created = created_at or _now_iso()
    report = dict(base)
    meta = dict(report.get("meta") or {})
    meta.setdefault("sessionId", session_id)
    if user_id:
        meta.setdefault("userId", user_id)
    meta.setdefault("createdAt", created)
    meta["reportId"] = rid
    report["meta"] = meta
    report["id"] = rid
    report["reportId"] = rid
    report["created_at"] = created
    return report


def _serialize_item(it):
    if it is None: return None
    return {
        "id": it.id,
        "domain": it.domain,
        "type": it.type,  # "MCQ"|"SJT"|"SR"|"OPEN"
        "prompt": getattr(it, "prompt", None) or getattr(it, "stem", None) or getattr(it, "text", None),
        "options": getattr(it, "options", None) or getattr(it, "choices", None),
        "image_url": getattr(it, "image_url", None),
        "alt": getattr(it, "alt", None),
    }

# ---- Health ----
@app.get("/health")
def health():
    return {
        "llm_backend": os.getenv("LLM_BACKEND", "none"),
        "use_llm_open": os.getenv("USE_LLM_OPEN", "0"),
        "azure_config_present": all(os.getenv(k) for k in [
            "AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_API_VERSION","AZURE_OPENAI_DEPLOYMENT"
        ])
    }

# ---- Low-level session endpoints (already used) ----
@app.post("/session/start")
def start(req: StartReq):
    sid = str(uuid.uuid4())
    if req.llm != "none":
        os.environ["USE_LLM_OPEN"] = "1"
        os.environ["LLM_BACKEND"] = req.llm
        if req.llm == "azure":
            ok = all(os.getenv(k) for k in [
                "AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_API_VERSION","AZURE_OPENAI_DEPLOYMENT"
            ])
            if not ok:
                raise HTTPException(500, "Azure LLM requested but AZURE_* env variables are missing on the server.")
    os.environ["RUN_ID"] = f"web_{int(time.time())}"
    sess = AdaptiveSession(run_type=("long" if req.run == "long" else "short"))
    SESS[sid] = sess
    # do NOT advance here for the /api/test contract; NEXT will serve items
    NEXT_CACHE[sid] = _serialize_item(sess.next_item())
    started_at = _now_iso()
    SESSION_INFO[sid] = {"user_id": req.user_id, "run": req.run, "started_at": started_at}
    if req.user_id:
        record_active_session(
            sid,
            {
                "sessionId": sid,
                "userId": req.user_id,
                "run": req.run,
                "startedAt": started_at,
                "lastUpdated": started_at,
                "lastItem": 0,
            },
        )
    return {"session_id": sid, "item": NEXT_CACHE[sid]}  # kept for backward compatibility

@app.post("/session/{sid}/answer")
def answer(sid: str, req: AnswerReq):
    sess = SESS.get(sid)
    if not sess: raise HTTPException(404, "session not found")
    val = req.value
    try: val = int(val)
    except Exception: pass
    sess.answer_current(Answer(item_id=req.item_id, value=val, rt_sec=(req.rt_ms/1000.0 if req.rt_ms else None)))
    nxt = sess.next_item()
    return {"done": nxt is None, "item": _serialize_item(nxt)}

@app.get("/session/{sid}/report")
def report(sid: str):
    stored = find_report_by_session(sid)
    if stored:
        return stored
    sess = SESS.get(sid)
    if not sess:
        raise HTTPException(404, "session not found")
    info = SESSION_INFO.get(sid, {})
    res = _serialize_result(sess.finalize())
    return _decorate_report(res, session_id=sid, user_id=info.get("user_id"))

@app.get("/session/{sid}/report/html")
def report_html_endpoint(sid: str):
    from skill_core.report_html import export_report_html
    import tempfile, json as _json
    stored = find_report_by_session(sid)
    if stored:
        d = stored
    else:
        sess = SESS.get(sid)
        if not sess:
            raise HTTPException(404, "session not found")
        info = SESSION_INFO.get(sid, {})
        res = _serialize_result(sess.finalize())
        d = _decorate_report(res, session_id=sid, user_id=info.get("user_id"))
    with tempfile.NamedTemporaryFile("w+", suffix=".html", delete=False, encoding="utf-8") as f:
        export_report_html(d, f.name)
        f.seek(0)
        return {"html": f.read()}

# ---- Frontend-friendly wrappers (your contract) ----
@app.get("/api/test/next")
def test_next(session_id: str):
    sess = SESS.get(session_id)
    if not sess: raise HTTPException(404, "session not found")
    item = NEXT_CACHE.pop(session_id, None)
    if item is None:
        item = _serialize_item(sess.next_item())
    return {"item": item}

@app.post("/api/test/answer")
def test_answer(payload: FEAnswer = Body(...)):
    sess = SESS.get(payload.session_id)
    if not sess: raise HTTPException(404, "session not found")
    rt_ms = None
    if payload.started_at is not None and payload.submitted_at is not None:
        try:
            rt_ms = max(0, int((payload.submitted_at - payload.started_at) * 1000))
        except Exception:
            rt_ms = None
    val = payload.answer
    try: val = int(val)
    except Exception: pass
    sess.answer_current(Answer(item_id=payload.item_id, value=val, rt_sec=(rt_ms/1000.0 if rt_ms else None)))
    NEXT_CACHE[payload.session_id] = _serialize_item(sess.next_item())
    meta = SESSION_INFO.get(payload.session_id, {})
    if meta.get("user_id"):
        update_active_session(
            payload.session_id,
            {
                "lastUpdated": _now_iso(),
                "lastItem": getattr(sess, "_step", None),
            },
        )
    return {"ok": True, "next_available": NEXT_CACHE[payload.session_id] is not None}

@app.post("/api/test/finish")
def test_finish(payload: FEFinish):
    sess = SESS.get(payload.session_id)
    if not sess: raise HTTPException(404, "session not found")
    info = SESSION_INFO.get(payload.session_id, {})
    res = _serialize_result(sess.finalize())
    report = _decorate_report(
        res,
        session_id=payload.session_id,
        user_id=info.get("user_id"),
    )
    metadata = {
        "sessionId": payload.session_id,
        "userId": info.get("user_id"),
        "createdAt": report["created_at"],
        "run": info.get("run") or report.get("run_type"),
        "kind": info.get("run") or report.get("run_type"),
        "summary": report.get("summary"),
    }
    save_report(report["id"], report, metadata)
    if info.get("user_id"):
        clear_active_session(payload.session_id)
    NEXT_CACHE.pop(payload.session_id, None)
    SESS.pop(payload.session_id, None)
    SESSION_INFO.pop(payload.session_id, None)
    return report


@app.get("/reports/{report_id}")
def get_report(report_id: str):
    report = load_report(report_id)
    if not report:
        raise HTTPException(404, "report not found")
    return report


@app.post("/results/{report_id}/plan")
def create_plan(report_id: str, force: bool = Query(False, description="Regenerate even if cached")):
    report = load_report(report_id)
    if not report:
        raise HTTPException(404, "result not found")

    existing = report.get("plan") if isinstance(report, dict) else None
    if existing and not force:
        return {"result_id": report_id, "plan": existing}

    cfg = load_config()
    plan = generate_plan(report, cfg)
    if isinstance(report, dict):
        report["plan"] = plan
        meta = report.get("meta") or {}
        metadata = {
            "sessionId": meta.get("sessionId"),
            "userId": meta.get("userId"),
            "createdAt": report.get("created_at") or meta.get("createdAt"),
            "run": meta.get("run") or meta.get("kind") or report.get("run_type"),
            "kind": meta.get("run") or meta.get("kind") or report.get("run_type"),
            "summary": report.get("summary"),
        }
        save_report(report_id, report, metadata)
    return {"result_id": report_id, "plan": plan}


@app.get("/results/{report_id}/audit.json")
def get_audit_json(report_id: str):
    if not AUDIT_EXPORT_ENABLED:
        raise HTTPException(404, "audit export disabled")

    report = load_report(report_id)
    if not report:
        raise HTTPException(404, "result not found")

    events = report.get("audit_events") if isinstance(report, dict) else None
    payload = audit_to_json(events or [])
    return {"result_id": report_id, **payload}


@app.get("/results/{report_id}/audit.csv")
def get_audit_csv(report_id: str):
    if not AUDIT_EXPORT_ENABLED:
        raise HTTPException(404, "audit export disabled")

    report = load_report(report_id)
    if not report:
        raise HTTPException(404, "result not found")

    events = report.get("audit_events") if isinstance(report, dict) else None
    body = audit_to_csv(events or [])
    filename = f"{report_id}_audit.csv"
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )


@app.delete("/reports/{report_id}")
def delete_report_endpoint(report_id: str):
    ok = delete_report(report_id)
    if not ok:
        raise HTTPException(404, "report not found")
    return {"ok": True}


@app.get("/users/{user_id}/reports")
def list_reports(user_id: str):
    reports = list_reports_for_user(user_id)
    return {"reports": reports}


@app.get("/users/{user_id}/sessions/active")
def list_active_sessions(user_id: str):
    sessions = active_sessions_for_user(user_id)
    return {"sessions": sessions}
