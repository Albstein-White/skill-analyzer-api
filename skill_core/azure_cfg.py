# skill_core/azure_cfg.py
from __future__ import annotations
import os, json, pathlib
from dataclasses import dataclass
from openai import AzureOpenAI

@dataclass(frozen=True)
class AzureSettings:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str

def _from_env() -> dict[str, str]:
    return {
        "endpoint":   os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "api_key":    os.getenv("AZURE_OPENAI_API_KEY", ""),
        "api_version":os.getenv("AZURE_OPENAI_API_VERSION", ""),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
    }

def _from_json(path: str = ".azure_config.json") -> dict[str, str]:
    p = pathlib.Path(path)
    if not p.exists(): return {}
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        return {
            "endpoint":   str(j.get("endpoint","")),
            "api_key":    str(j.get("api_key","")),
            "api_version":str(j.get("api_version","")),
            "deployment": str(j.get("deployment","")),
        }
    except Exception:
        return {}

def settings() -> AzureSettings:
    cfg = _from_env()
    if not all(cfg.values()):
        j = _from_json()
        for k,v in j.items():
            if not cfg.get(k): cfg[k] = v
    missing = [k for k,v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Azure OpenAI not configured. Missing: {', '.join(missing)}")
    return AzureSettings(
        endpoint=cfg["endpoint"],
        api_key=cfg["api_key"],
        deployment=cfg["deployment"],
        api_version=cfg["api_version"],
    )

def client() -> AzureOpenAI:
    s = settings()
    return AzureOpenAI(
        azure_endpoint=s.endpoint,
        api_key=s.api_key,
        api_version=s.api_version,
    )
