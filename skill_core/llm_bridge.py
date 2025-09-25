from __future__ import annotations
import json, os, time
from typing import Dict, Any
from .azure_cfg import client as azure_client, settings as azure_settings
from .heuristics import heuristic_open_score

def backend_in_use() -> str:
    b = (os.getenv("LLM_BACKEND") or "").lower()
    return b if b in ("azure", "ollama") else "none"

def _grade_azure(prompt_stub: str, user_answer: str) -> str:
    s = azure_settings(); cli = azure_client()
    system = ("You are a strict rubric-based grader. "
              "Return ONLY compact JSON with keys: overall, structure, metrics, actions, clarity. "
              "Values must be floats in [0,1]. No explanations.")
    user = f"{(prompt_stub or '').strip()}\n\nAnswer:\n{(user_answer or '').strip()}"
    resp = cli.chat.completions.create(
        model=s.deployment, messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.0, max_tokens=200, top_p=1.0,
    )
    return resp.choices[0].message.content or "{}"

def score_open(item_id: str, answer: str, prompt_stub: str | None = None) -> float:
    t0 = time.time()
    backend = backend_in_use()
    raw_json: Dict[str, Any] | None = None
    try:
        if backend == "azure":
            raw = _grade_azure(prompt_stub or "", answer or "")
            raw_json = json.loads(raw); overall = float(raw_json.get("overall", 0.0))
        else:
            overall = heuristic_open_score(answer or "")
    except Exception as e:
        overall = heuristic_open_score(answer or ""); raw_json = {"error": str(e)}
    A = float(os.getenv("OPEN_CAL_A", "1.0")); B = float(os.getenv("OPEN_CAL_B", "0.0"))
    s = max(0.0, min(1.0, A*overall + B))
    try:
        log = {
            "ts": round(time.time(), 3),
            "run_id": os.getenv("RUN_ID", ""),
            "profile": os.getenv("PROFILE", ""),
            "item": item_id,
            "backend": backend,
            "prompt": (prompt_stub or "")[:800],
            "answer": (answer or "")[:1200],
            "raw": raw_json,
            "score": s,
            "rt_ms": int((time.time()-t0)*1000),
        }
        with open("llm_open_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return s
