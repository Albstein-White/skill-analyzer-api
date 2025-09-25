@'
import os, json, pathlib

def load_azure():
    cfg = {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", ""),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
    }
    if all(cfg.values()):
        return cfg
    p = pathlib.Path(".azure_config.json")
    if p.exists():
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            return {
                "endpoint": j.get("endpoint",""),
                "api_key": j.get("api_key",""),
                "api_version": j.get("api_version",""),
                "deployment": j.get("deployment",""),
            }
        except Exception:
            pass
    return cfg
'@ | Set-Content -Encoding utf8 .\skill_core\azure_cfg.py
