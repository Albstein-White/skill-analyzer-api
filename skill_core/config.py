from __future__ import annotations
import math
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


def _env_float(name: str, default: float | None) -> float | None:
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

CAP_SHORT: int = 48
CAP_LONG: int = 160

SHORT_PASS: dict[str, int] = {"need": 2, "window": 3}
PASS_RULE_SHORT = SHORT_PASS
PASS_RULE_LONG  = {"need": 3, "window": 4}
SHORT_DEMOTE: dict[str, int] = {"need": 2, "window": 3}
DEMOTE_RULE     = {"need": 2, "window": 3}

TIER_NAMES: tuple[str, ...] = ("F", "D", "C", "B", "A", "S", "SS", "GOD")
SHORT_TIER_CAP: str = "A"
GOD_MIN_NORM: float = 9.85
GOD_MAX_SE: float = 0.20
GOD_REQ_LEVEL: int = 2
GOD_MIN_L2_SEEN_DEFAULT: int = 4
GOD_MIN_L2_SEEN: int = GOD_MIN_L2_SEEN_DEFAULT
GOD_MIN_L2_ACC: float = 0.80
GOD_MIN_OPEN_DEFAULT: int = 2
GOD_MIN_OPEN: int = GOD_MIN_OPEN_DEFAULT
GOD_MIN_R1: float = 0.80
GOD_MIN_R2: float = 0.75

SE_TARGET_SHORT: float = 0.35
SE_TARGET_LONG: float = 0.25

SHORT_OBJ_MIN: int = 3
SHORT_OBJ_MAX: int = 5
SHORT_EXTRA_TOTAL: int = 8  # +2 objectives for four domains
SHORT_SR_PER_DOMAIN: int = 1
SHORT_LEVELS: tuple[int, int, int] = (-1, 0, 1)
SHORT_START_LEVEL: int = 0
OBJ_MIN_SHORT: int = SHORT_OBJ_MIN
OBJ_MAX_SHORT: int = SHORT_OBJ_MAX
OBJ_MIN_LONG: int = 8
OBJ_MAX_LONG: int = 16

LEVEL_MIN: int = -2
LEVEL_MAX: int = 2

GLOBAL_STEP_CAP: int = CAP_LONG

OPEN_ENABLED_LONG: bool = True
OPEN_GATE1_MIN_OBJ: int = 4
OPEN_GATE1_SE_MAX: float = 0.60
OPEN_GATE2_SE_MAX: float = 0.45
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
OPEN_FALLBACK_AFTER_OBJ: int = 12
OPEN_FALLBACK_MIN_STABLE: int = 1
OPEN_DEBUG_REASON: bool = True

SR_PER_DOMAIN_SHORT: int = SHORT_SR_PER_DOMAIN
SR_PER_DOMAIN_LONG: int = 2
SR_START_SHIFT: dict[str, int] = {"low": -1, "mid": 0, "high": 1}

BANK_MIN_PER_BUCKET_OBJ: int = 20
BANK_MIN_PER_BUCKET_OPEN: int = 10
BANK_EXPECT_VARIANT_GROUP: bool = True
BANK_AUDIT_ALLOW_WARN: bool = False

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
TEST_MODE: bool = False
TEST_OPEN_FULL: bool = False
TEST_GOD_MAX_SE: float | None = None
TEST_GOD_MIN_NORM: float | None = None
TEST_GOD_MIN_L2_SEEN: int | None = None
TEST_GOD_MIN_L2_ACC: float | None = None
TEST_GOD_MIN_OPEN: int | None = None
TEST_GOD_RUBRIC0: float | None = None
TEST_GOD_RUBRIC1: float | None = None
OPEN_RESERVE_FORCE: bool = False
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
TEST_MODE = _env_bool("TEST_MODE", TEST_MODE)
TEST_OPEN_FULL = _env_bool("TEST_OPEN_FULL", TEST_OPEN_FULL)
TEST_GOD_MAX_SE = _env_float("TEST_GOD_MAX_SE", TEST_GOD_MAX_SE)
TEST_GOD_MIN_NORM = _env_float("TEST_GOD_MIN_NORM", TEST_GOD_MIN_NORM)
TEST_GOD_MIN_L2_SEEN = _env_int("TEST_GOD_MIN_L2_SEEN", TEST_GOD_MIN_L2_SEEN or 0)
TEST_GOD_MIN_L2_ACC = _env_float("TEST_GOD_MIN_L2_ACC", TEST_GOD_MIN_L2_ACC)
TEST_GOD_MIN_OPEN = _env_int("TEST_GOD_MIN_OPEN", TEST_GOD_MIN_OPEN or 0)
TEST_GOD_RUBRIC0 = _env_float("TEST_GOD_RUBRIC0", TEST_GOD_RUBRIC0)
TEST_GOD_RUBRIC1 = _env_float("TEST_GOD_RUBRIC1", TEST_GOD_RUBRIC1)
if TEST_GOD_MIN_L2_SEEN == 0 and TEST_GOD_MIN_L2_SEEN is not None:
    TEST_GOD_MIN_L2_SEEN = None
if TEST_GOD_MIN_OPEN == 0 and TEST_GOD_MIN_OPEN is not None:
    TEST_GOD_MIN_OPEN = None
AUDIT_EXPORT_ENABLED = _env_bool("AUDIT_EXPORT_ENABLED", AUDIT_EXPORT_ENABLED)
BANK_AUDIT_ALLOW_WARN = _env_bool("BANK_AUDIT_ALLOW_WARN", BANK_AUDIT_ALLOW_WARN)

if STAGING_PROFILE and TEST_MODE and TEST_OPEN_FULL:
    OPEN_RESERVE_FORCE = True


def god_thresholds(cfg: object | None = None) -> dict[str, float | int]:
    """Return the effective GOD gate thresholds with staging overrides."""

    module = cfg if cfg is not None else globals()

    def _value(name: str):
        return module[name] if isinstance(module, dict) else getattr(module, name)

    thresholds = {
        "min_norm": float(_value("GOD_MIN_NORM")),
        "max_se": float(_value("GOD_MAX_SE")),
        "min_l2_seen": int(_value("GOD_MIN_L2_SEEN")),
        "min_l2_acc": float(_value("GOD_MIN_L2_ACC")),
        "min_open": int(_value("GOD_MIN_OPEN")),
        "rubric0": float(_value("GOD_MIN_R1")),
        "rubric1": float(_value("GOD_MIN_R2")),
    }

    staging = bool(_value("STAGING_PROFILE"))
    test_mode = bool(_value("TEST_MODE"))

    if staging and test_mode:
        min_norm = _value("TEST_GOD_MIN_NORM")
        max_se = _value("TEST_GOD_MAX_SE")
        min_l2_seen = _value("TEST_GOD_MIN_L2_SEEN")
        min_l2_acc = _value("TEST_GOD_MIN_L2_ACC")
        min_open = _value("TEST_GOD_MIN_OPEN")
        rubric0 = _value("TEST_GOD_RUBRIC0")
        rubric1 = _value("TEST_GOD_RUBRIC1")

        if min_norm is not None:
            thresholds["min_norm"] = float(min_norm)
        if max_se is not None:
            thresholds["max_se"] = float(max_se)
        if min_l2_seen is not None:
            thresholds["min_l2_seen"] = int(min_l2_seen)
        if min_l2_acc is not None:
            thresholds["min_l2_acc"] = float(min_l2_acc)
        if min_open is not None:
            thresholds["min_open"] = int(min_open)
        if rubric0 is not None:
            thresholds["rubric0"] = float(rubric0)
        if rubric1 is not None:
            thresholds["rubric1"] = float(rubric1)

        test_min_open = min_open
        test_min_l2_seen = min_l2_seen
    else:
        test_min_open = None
        test_min_l2_seen = None

    if staging and test_mode:
        min_open_val = thresholds["min_open"]
        if test_min_open is not None and min_open_val > 0:
            try:
                from .question_bank import DOMAINS as _DOMAINS  # type: ignore import-not-found
                domain_count = max(1, len(_DOMAINS))
            except Exception:  # pragma: no cover - defensive guard
                domain_count = 1
            if test_min_open > GOD_MIN_OPEN_DEFAULT and test_min_open >= domain_count:
                thresholds["min_open"] = max(
                    1,
                    int(math.ceil(float(test_min_open) / float(domain_count))),
                )

        min_l2_seen_val = thresholds["min_l2_seen"]
        if test_min_l2_seen is not None and min_l2_seen_val > 0:
            try:
                from .question_bank import DOMAINS as _DOMAINS  # type: ignore import-not-found
                domain_count = max(1, len(_DOMAINS))
            except Exception:  # pragma: no cover - defensive guard
                domain_count = 1
            if test_min_l2_seen > GOD_MIN_L2_SEEN_DEFAULT and test_min_l2_seen >= domain_count:
                thresholds["min_l2_seen"] = max(
                    1,
                    int(math.ceil(float(test_min_l2_seen) / float(domain_count))),
                )

    return thresholds

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
