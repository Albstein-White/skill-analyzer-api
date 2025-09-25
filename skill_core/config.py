
from __future__ import annotations
import os, json, pathlib, random
def _env_true(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1","true","yes","on")
def load_config() -> dict:
    cfg = {}
    p = pathlib.Path("config.json")
    if p.exists():
        try: cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception: cfg = {}
    e = os.environ
    if e.get("USE_LLM_OPEN"): cfg["USE_LLM_OPEN"] = _env_true("USE_LLM_OPEN")
    if e.get("LLM_BACKEND"): cfg["LLM_BACKEND"] = e.get("LLM_BACKEND")
    if e.get("OLLAMA_HOST"): cfg["OLLAMA_HOST"] = e.get("OLLAMA_HOST")
    if e.get("OLLAMA_MODEL"): cfg["OLLAMA_MODEL"] = e.get("OLLAMA_MODEL")
    for k in ("AZURE_OAI_ENDPOINT","AZURE_OAI_API_VERSION","AZURE_OAI_API_KEY","AZURE_OAI_DEPLOY_SCORING"):
        if e.get(k): cfg[k] = e.get(k)
    if e.get("SEED"): cfg["SEED"] = int(e.get("SEED"))
    return cfg
def get_backend(cfg: dict) -> str|None:
    if not cfg.get("USE_LLM_OPEN"): return None
    b = (cfg.get("LLM_BACKEND") or "").lower().strip()
    return b if b in ("azure","ollama") else None
def seed_rng(cfg: dict):
    s = cfg.get("SEED")
    if s is not None:
        random.seed(int(s))
