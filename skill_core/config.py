from __future__ import annotations
import os, json, pathlib, random


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


RASCH_PRIOR_VAR: float = 1.0
RASCH_ETA: float = 1.0
MIN_LEVEL: int = -2
MAX_LEVEL: int = 2

CAP_SHORT: int = 104
CAP_LONG: int = 160

PASS_RULE_SHORT = {"need": 2, "window": 3}
PASS_RULE_LONG  = {"need": 3, "window": 4}
DEMOTE_RULE     = {"need": 2, "window": 3}

SE_TARGET_SHORT: float = 0.35
SE_TARGET_LONG: float = 0.25

OBJ_MIN_SHORT: int = 4
OBJ_MAX_SHORT: int = 12
OBJ_MIN_LONG: int = 8
OBJ_MAX_LONG: int = 16

LEVEL_MIN: int = -2
LEVEL_MAX: int = 2

GLOBAL_STEP_CAP: int = CAP_LONG

OPEN_ENABLED_LONG: bool = True
OPEN_GATE1_MIN_OBJ: int = 4
OPEN_GATE1_SE_MAX: float = 0.45
OPEN_GATE2_SE_MAX: float = 0.30
OPEN_GATE2_MIN_R1: float = 0.60
OPEN_LEVELS: tuple[int, int, int] = (-1, 0, 1)
A_OPEN: float = 0.50
ETA_OPEN: float = 0.10
DELTA_OPEN_MAX: float = 0.30
OPEN_INFO_CAP_RATIO: float = 0.20
OPEN_MIN_WORDS: int = 80
OPEN_MIN_RUBRIC: float = 0.60
OPEN_LATENCY_P10_MS: int = 1500
OPEN_Z_SHIFT: float = 0.50

SR_PER_DOMAIN_SHORT: int = 1
SR_PER_DOMAIN_LONG: int = 2
SR_START_SHIFT: dict[str, int] = {"low": -1, "mid": 0, "high": 1}

BANK_MIN_PER_BUCKET_OBJ: int = 20
BANK_MIN_PER_BUCKET_OPEN: int = 10
BANK_EXPECT_VARIANT_GROUP: bool = True

AUDIT_EXPORT_ENABLED: bool = True

PLAN_ENABLED: bool = True
PLAN_LONG_ONLY: bool = True
PLAN_FOCUS_MAX: int = 3
PLAN_BULLETS_MIN: int = 3
PLAN_BULLETS_MAX: int = 5
PLAN_MIN_ITEMS_LONG: int = 90
PLAN_SE_TARGET_LONG: float = SE_TARGET_LONG
PLAN_LLM_ENABLED: bool = False
PLAN_LATENCY_BUCKETS: tuple[float, float] = (0.25, 0.75)

DEBUG_TRACE: bool = False
DEBUG_SEED: int | None = None
STAGING_PROFILE: bool = False
TRACE_FIELDS: tuple[str, ...] = (
    "domain",
    "item_id",
    "type",
    "level",
    "b",
    "correct_or_r",
    "theta_before",
    "theta_after",
    "se",
    "info_gain",
)
# // env overrides for staging/ops; defaults remain conservative.
PLAN_ENABLED = _env_bool("PLAN_ENABLED", PLAN_ENABLED)
OPEN_ENABLED_LONG = _env_bool("OPEN_ENABLED_LONG", OPEN_ENABLED_LONG)
CAP_SHORT = _env_int("CAP_SHORT", CAP_SHORT)
CAP_LONG = _env_int("CAP_LONG", CAP_LONG)
DEBUG_TRACE = _env_bool("DEBUG_TRACE", False)
DEBUG_SEED = os.getenv("DEBUG_SEED", None)
STAGING_PROFILE = _env_bool("STAGING_PROFILE", False)
AUDIT_EXPORT_ENABLED = _env_bool("AUDIT_EXPORT_ENABLED", AUDIT_EXPORT_ENABLED)

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
