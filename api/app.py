from __future__ import annotations
from fastapi import FastAPI, HTTPException, Body
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

SESS: dict[str, AdaptiveSession] = {}
NEXT_CACHE: dict[str, t.Any] = {}  # sid -> serialized next item

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
    sess = SESS.get(sid)
    if not sess: raise HTTPException(404, "session not found")
    res = sess.finalize()
    return json.loads(json.dumps(res, default=lambda o: getattr(o, "__dict__", o)))

@app.get("/session/{sid}/report/html")
def report_html_endpoint(sid: str):
    from skill_core.report_html import export_report_html
    import tempfile, json as _json
    sess = SESS.get(sid)
    if not sess: raise HTTPException(404, "session not found")
    res = sess.finalize()
    d = _json.loads(_json.dumps(res, default=lambda o: getattr(o,"__dict__",o)))
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
    return {"ok": True, "next_available": NEXT_CACHE[payload.session_id] is not None}

@app.post("/api/test/finish")
def test_finish(payload: FEFinish):
    sess = SESS.get(payload.session_id)
    if not sess: raise HTTPException(404, "session not found")
    res = sess.finalize()
    # optional: cleanup caches
    NEXT_CACHE.pop(payload.session_id, None)
    return json.loads(json.dumps(res, default=lambda o: getattr(o, "__dict__", o)))
