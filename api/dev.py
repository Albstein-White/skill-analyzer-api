from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, Optional, Sequence

from fastapi import APIRouter, FastAPI, HTTPException, Query

import skill_core.engine as engine_mod
from skill_core.scoring import score_item
from skill_core.types import Answer
from skill_core import config as cfg
import skill_core.policy as policy_mod
from skill_core.engine import AdaptiveSession

router = APIRouter()

# Standalone FastAPI application for developer utilities. This allows
# `from api.dev import app` to work in diagnostics without depending on the
# staging-only wiring inside `api.app`.
app = FastAPI(title="skill-analyzer-dev")


def _open_pass_response() -> str:
    sentence = (
        "Deliver evidence-based analysis with clear transitions, cite quantitative benchmarks, outline weekly review cadences, "
        "and close with measurable decision gates to sustain 90 percent accuracy."
    )
    words = sentence.split()
    payload: list[str] = []
    while len(payload) < 150:
        payload.extend(words)
    return " ".join(payload[:150])


def _candidate_indices(item: Any) -> Sequence[int]:
    indices: set[int] = set()
    for attr in ("options", "choices"):
        opts = getattr(item, attr, None)
        if isinstance(opts, Sequence) and not isinstance(opts, (str, bytes)):
            indices.update(range(len(opts)))
    keys = getattr(item, "keys", None) or getattr(item, "sjt_keys", None)
    if isinstance(keys, dict):
        for key in keys.keys():
            try:
                indices.add(int(key))
            except Exception:
                continue
    for attr in ("correct", "best_index", "best", "key_best", "good_index", "good", "key_good"):
        val = getattr(item, attr, None)
        if val is None:
            continue
        try:
            indices.add(int(val))
        except Exception:
            continue
    if not indices:
        indices.update(range(4))
    return tuple(sorted(idx for idx in indices if idx >= 0))


def _best_objective_answer(item: Any) -> str:
    best_idx: Optional[int] = None
    best_credit = -1.0
    for idx in _candidate_indices(item):
        ans = Answer(item_id=getattr(item, "id", ""), value=str(idx))
        try:
            credit, _ = score_item(item, ans)
        except Exception:
            credit = 0.0
        if credit > best_credit:
            best_credit = credit
            best_idx = idx
    return str(best_idx if best_idx is not None else 0)


def _pass_value(item: Any) -> str:
    item_type = str(getattr(item, "type", "")).upper()
    if item_type in {"MCQ", "SJT"}:
        return _best_objective_answer(item)
    if item_type == "OPEN":
        return _open_pass_response()
    if item_type == "SR":
        return "3"
    return "1"


def _fail_value(item: Any) -> str:
    item_type = str(getattr(item, "type", "")).upper()
    if item_type in {"MCQ", "SJT"}:
        best = _best_objective_answer(item)
        indices = list(_candidate_indices(item)) or [0]
        try:
            best_int = int(best)
        except Exception:
            best_int = 0
        for idx in indices:
            if idx != best_int:
                return str(idx)
        return str((best_int + 1) % max(len(indices), 1) if indices else 1)
    if item_type == "SR":
        return "0"
    if item_type == "OPEN":
        return "ok"
    return "1"


def _apply_seed(seed: Optional[int]) -> Optional[object]:
    if seed is None:
        return None
    prev_state = random.getstate()
    random.seed(int(seed))
    return prev_state


def simulate(run: str, mode: str, seed: Optional[int] = None) -> Dict[str, Any]:
    run_lower = run.lower()
    mode_lower = mode.lower()
    if run_lower not in {"short", "long"}:
        raise ValueError("run must be 'short' or 'long'")
    if mode_lower == "god":
        mode_lower = "pass"
    if mode_lower not in {"pass", "fail"}:
        raise ValueError("mode must be 'pass' or 'fail'")

    prev_state = _apply_seed(seed)
    prev_debug_seed = os.environ.get("DEBUG_SEED")
    if seed is not None:
        os.environ["DEBUG_SEED"] = str(seed)
    else:
        os.environ.pop("DEBUG_SEED", None)

    try:
        session = AdaptiveSession(run_type=run_lower)
    finally:
        if prev_state is not None:
            random.setstate(prev_state)
        if prev_debug_seed is None:
            os.environ.pop("DEBUG_SEED", None)
        else:
            os.environ["DEBUG_SEED"] = prev_debug_seed

    if seed is not None and hasattr(session, "policy") and hasattr(session.policy, "rng"):
        session.policy.rng.seed(int(seed))

    steps = 0
    while not session.done:
        item = session.next_item()
        if item is None:
            break
        item_type = str(getattr(item, "type", "")).upper()
        value = _pass_value(item) if mode_lower == "pass" else _fail_value(item)
        rt_sec = 2.6 if item_type == "OPEN" else 2.0
        answer = Answer(item_id=getattr(item, "id", ""), value=value, rt_sec=rt_sec)
        session.answer_current(answer)
        steps += 1

    result = session.finalize()
    if isinstance(result, dict):
        summary = result.get("summary", {}) or {}
        domain_scores = result.get("domain_scores", []) or []
    else:
        summary = getattr(result, "summary", {}) or {}
        domain_scores = getattr(result, "domain_scores", []) or []

    tiers: list[str] = []
    reasons: Dict[str, list[str]] = {}
    for score in domain_scores:
        if isinstance(score, dict):
            tier_val = str(score.get("tier", ""))
            domain = str(score.get("domain", ""))
            tier_reasons = score.get("tier_reasons")
            if tier_reasons is None:
                gate_meta = score.get("god_gate")
                if isinstance(gate_meta, dict):
                    tier_reasons = gate_meta.get("reasons")
        else:
            tier_val = str(getattr(score, "tier", ""))
            domain = str(getattr(score, "domain", ""))
            tier_reasons = getattr(score, "tier_reasons", None)
            if tier_reasons is None:
                gate_meta = getattr(score, "god_gate", None)
                if isinstance(gate_meta, dict):
                    tier_reasons = gate_meta.get("reasons")

        tiers.append(tier_val)
        if domain:
            reasons[domain] = list(tier_reasons or [])

    return {
        "steps": int(summary.get("steps_used", steps)),
        "cap": summary.get("cap"),
        "sr_used": int(summary.get("sr_used", getattr(session, "sr_used", 0))),
        "open_used": int(summary.get("open_used", getattr(session, "open_used", 0))),
        "effective_steps": int(
            summary.get("effective_steps", getattr(session, "effective_steps", steps))
        ),
        "extra_steps_exempt": int(
            summary.get(
                "extra_steps_exempt", getattr(session, "extra_steps_exempt", 0)
            )
        ),
        "tiers": tiers,
        "reasons": reasons,
        "stop_reason": summary.get("stop_reason") or getattr(session, "stop_reason", None),
        "ff_win_obj_len_short": summary.get("ff_win_obj_len_short"),
        "ff_win_obj_len_long": summary.get("ff_win_obj_len_long"),
        "god_probe": summary.get("god_probe"),
        "god_probe_enabled": bool(
            summary.get("god_probe_enabled", getattr(session, "god_probe_enabled", False))
        ),
        "open_first_tier": summary.get("open_first_tier"),
        "first_open_was_below_ss": summary.get("first_open_was_below_ss"),
        "open_floor_applied": summary.get("open_floor_applied"),
        "per_domain": summary.get("per_domain"),
        "totals": summary.get("totals"),
        "obj_total": summary.get("obj_total"),
        "per_domain_obj_count": summary.get("per_domain_obj_count"),
        "long_early_stop_enabled": bool(summary.get("long_early_stop_enabled", False)),
    }


def _is_truthy(value: Optional[object]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@router.get("/dev/run")
def dev_run(
    run: str = Query("short"),
    seed: Optional[int] = Query(None),
    mode: str = Query("pass"),
    test_mode_flag: Optional[str] = Query(None, alias="TEST_MODE"),
    test_min_norm: Optional[float] = Query(None, alias="TEST_GOD_MIN_NORM"),
    test_max_se: Optional[float] = Query(None, alias="TEST_GOD_MAX_SE"),
    test_min_l2_seen: Optional[int] = Query(None, alias="TEST_GOD_MIN_L2_SEEN"),
    test_min_l2_acc: Optional[float] = Query(None, alias="TEST_GOD_MIN_L2_ACC"),
    test_min_open: Optional[int] = Query(None, alias="TEST_GOD_MIN_OPEN"),
    test_rubric0: Optional[float] = Query(None, alias="TEST_GOD_RUBRIC0"),
    test_rubric1: Optional[float] = Query(None, alias="TEST_GOD_RUBRIC1"),
    test_rubric_primary: Optional[float] = Query(
        None, alias="TEST_GOD_MIN_OPEN_RUBRIC"
    ),
    test_rubric_secondary: Optional[float] = Query(
        None, alias="TEST_GOD_MIN_OPEN_RUBRIC_SEC"
    ),
    test_open_full: Optional[str] = Query(None, alias="TEST_OPEN_FULL"),
    test_early_stop: Optional[str] = Query(None, alias="TEST_EARLY_STOP"),
    long_min_open_floor: Optional[int] = Query(None, alias="LONG_MIN_OBJ_FOR_FIRST_OPEN"),
    long_min_obj_per_domain: Optional[int] = Query(
        None, alias="LONG_MIN_OBJ_PER_DOMAIN"
    ),
    long_estop_disable: Optional[str] = Query(None, alias="LONG_ESTOP_DISABLE"),
    long_estop_min_obj: Optional[int] = Query(None, alias="LONG_ESTOP_MIN_OBJ"),
    long_estop_conf: Optional[float] = Query(None, alias="LONG_ESTOP_CONF"),
    prod_god_enable: Optional[str] = Query(None, alias="PROD_GOD_ENABLE"),
    prod_god_min_norm: Optional[float] = Query(None, alias="PROD_GOD_MIN_NORM"),
    prod_god_max_se: Optional[float] = Query(None, alias="PROD_GOD_MAX_SE"),
    prod_god_min_l2_seen: Optional[int] = Query(None, alias="PROD_GOD_MIN_L2_SEEN"),
    prod_god_min_l2_acc: Optional[float] = Query(None, alias="PROD_GOD_MIN_L2_ACC"),
    prod_god_min_open: Optional[int] = Query(None, alias="PROD_GOD_MIN_OPEN"),
    prod_god_rubric0: Optional[float] = Query(None, alias="PROD_GOD_RUBRIC0"),
    prod_god_rubric1: Optional[float] = Query(None, alias="PROD_GOD_RUBRIC1"),
) -> Dict[str, Any]:
    snapshots = {
        "TEST_MODE": cfg.TEST_MODE,
        "TEST_GOD_MIN_NORM": cfg.TEST_GOD_MIN_NORM,
        "TEST_GOD_MAX_SE": cfg.TEST_GOD_MAX_SE,
        "TEST_GOD_MIN_L2_SEEN": cfg.TEST_GOD_MIN_L2_SEEN,
        "TEST_GOD_MIN_L2_ACC": cfg.TEST_GOD_MIN_L2_ACC,
        "TEST_GOD_MIN_OPEN": cfg.TEST_GOD_MIN_OPEN,
        "TEST_GOD_RUBRIC0": getattr(cfg, "TEST_GOD_RUBRIC0", None),
        "TEST_GOD_RUBRIC1": getattr(cfg, "TEST_GOD_RUBRIC1", None),
        "TEST_GOD_MIN_OPEN_RUBRIC": getattr(
            cfg, "TEST_GOD_MIN_OPEN_RUBRIC", None
        ),
        "TEST_GOD_MIN_OPEN_RUBRIC_SEC": getattr(
            cfg, "TEST_GOD_MIN_OPEN_RUBRIC_SEC", None
        ),
        "TEST_OPEN_FULL": getattr(cfg, "TEST_OPEN_FULL", False),
        "OPEN_RESERVE_FORCE": getattr(cfg, "OPEN_RESERVE_FORCE", False),
        "TEST_EARLY_STOP": getattr(cfg, "TEST_EARLY_STOP", None),
        "LONG_MIN_OBJ_FOR_FIRST_OPEN": getattr(
            cfg, "LONG_MIN_OBJ_FOR_FIRST_OPEN", 8
        ),
        "POLICY_TEST_MODE": getattr(policy_mod, "TEST_MODE", False),
        "POLICY_TEST_OPEN_FULL": getattr(policy_mod, "TEST_OPEN_FULL", False),
        "POLICY_OPEN_RESERVE_FORCE": getattr(policy_mod, "OPEN_RESERVE_FORCE", False),
        "POLICY_TEST_EARLY_STOP": getattr(policy_mod, "TEST_EARLY_STOP", None),
        "POLICY_LONG_MIN_OBJ_FOR_FIRST_OPEN": getattr(
            policy_mod, "LONG_MIN_OBJ_FOR_FIRST_OPEN", getattr(cfg, "LONG_MIN_OBJ_FOR_FIRST_OPEN", 8)
        ),
        "LONG_MIN_OBJ_PER_DOMAIN": getattr(
            cfg, "LONG_MIN_OBJ_PER_DOMAIN", getattr(cfg, "LONG_OBJ_MIN", 8)
        ),
        "ENGINE_LONG_MIN_OBJ_PER_DOMAIN": getattr(
            engine_mod, "LONG_MIN_OBJ_PER_DOMAIN", getattr(cfg, "LONG_OBJ_MIN", 8)
        ),
        "POLICY_LONG_MIN_OBJ_PER_DOMAIN": getattr(
            policy_mod, "LONG_MIN_OBJ_PER_DOMAIN", getattr(cfg, "LONG_OBJ_MIN", 8)
        ),
        "LONG_OBJ_MIN": getattr(cfg, "LONG_OBJ_MIN", 8),
        "LONG_OBJ_MAX": getattr(cfg, "LONG_OBJ_MAX", 16),
        "ENGINE_LONG_OBJ_MIN": getattr(engine_mod, "LONG_OBJ_MIN", 8),
        "ENGINE_LONG_OBJ_MAX": getattr(engine_mod, "LONG_OBJ_MAX", 16),
        "POLICY_LONG_OBJ_MIN": getattr(policy_mod, "LONG_OBJ_MIN", 8),
        "POLICY_LONG_OBJ_MAX": getattr(policy_mod, "LONG_OBJ_MAX", 16),
        "LONG_EARLY_STOP_ENABLE": getattr(cfg, "LONG_EARLY_STOP_ENABLE", True),
        "ENGINE_LONG_EARLY_STOP_ENABLE": getattr(
            engine_mod, "LONG_EARLY_STOP_ENABLE", True
        ),
        "POLICY_LONG_EARLY_STOP_ENABLE": getattr(
            policy_mod, "LONG_EARLY_STOP_ENABLE", True
        ),
        "LONG_EARLY_SE": getattr(cfg, "LONG_EARLY_SE", 0.6),
        "ENGINE_LONG_EARLY_SE": getattr(
            engine_mod, "LONG_EARLY_SE", getattr(cfg, "LONG_EARLY_SE", 0.6)
        ),
        "POLICY_LONG_EARLY_SE": getattr(
            policy_mod, "LONG_EARLY_SE", getattr(cfg, "LONG_EARLY_SE", 0.6)
        ),
        "GOD_MIN_NORM": getattr(cfg, "GOD_MIN_NORM", 0.0),
        "GOD_MAX_SE": getattr(cfg, "GOD_MAX_SE", 0.0),
        "GOD_MIN_L2_SEEN": getattr(cfg, "GOD_MIN_L2_SEEN", 0),
        "GOD_MIN_L2_ACC": getattr(cfg, "GOD_MIN_L2_ACC", 0.0),
        "GOD_MIN_OPEN": getattr(cfg, "GOD_MIN_OPEN", 0),
        "PROD_GOD_ENABLE": getattr(cfg, "PROD_GOD_ENABLE", False),
        "GOD_PROBE_MIN_FIRST": getattr(cfg, "GOD_PROBE_MIN_FIRST", 0.85),
        "GOD_PROBE_MIN_SECOND": getattr(cfg, "GOD_PROBE_MIN_SECOND", 0.85),
        "GOD_PROBE_DIFFICULTY": getattr(cfg, "GOD_PROBE_DIFFICULTY", 1.0),
        "GOD_PROBE_EXEMPT_FROM_CAP": getattr(
            cfg, "GOD_PROBE_EXEMPT_FROM_CAP", True
        ),
        "GOD_MIN_R1": getattr(cfg, "GOD_MIN_R1", 0.80),
        "GOD_MIN_R2": getattr(cfg, "GOD_MIN_R2", 0.75),
        "PROD_GOD_RUBRIC0": getattr(
            cfg,
            "PROD_GOD_RUBRIC0",
            getattr(cfg, "GOD_PROBE_MIN_FIRST", 0.85),
        ),
        "PROD_GOD_RUBRIC1": getattr(
            cfg,
            "PROD_GOD_RUBRIC1",
            getattr(cfg, "GOD_PROBE_MIN_SECOND", 0.85),
        ),
        "ENGINE_GOD_MIN_NORM": getattr(engine_mod, "GOD_MIN_NORM", 0.0),
        "ENGINE_GOD_MAX_SE": getattr(engine_mod, "GOD_MAX_SE", 0.0),
        "ENGINE_GOD_MIN_L2_SEEN": getattr(engine_mod, "GOD_MIN_L2_SEEN", 0),
        "ENGINE_GOD_MIN_L2_ACC": getattr(engine_mod, "GOD_MIN_L2_ACC", 0.0),
        "ENGINE_GOD_MIN_OPEN": getattr(engine_mod, "GOD_MIN_OPEN", 0),
        "ENGINE_PROD_GOD_ENABLE": getattr(engine_mod, "PROD_GOD_ENABLE", False),
        "ENGINE_GOD_PROBE_MIN_FIRST": getattr(
            engine_mod, "GOD_PROBE_MIN_FIRST", 0.85
        ),
        "ENGINE_GOD_PROBE_MIN_SECOND": getattr(
            engine_mod, "GOD_PROBE_MIN_SECOND", 0.85
        ),
        "ENGINE_GOD_PROBE_DIFFICULTY": getattr(
            engine_mod, "GOD_PROBE_DIFFICULTY", 1.0
        ),
        "ENGINE_GOD_PROBE_EXEMPT_FROM_CAP": getattr(
            engine_mod, "GOD_PROBE_EXEMPT_FROM_CAP", True
        ),
        "ENGINE_PROD_GOD_RUBRIC0": getattr(
            engine_mod,
            "PROD_GOD_RUBRIC0",
            getattr(engine_mod, "GOD_PROBE_MIN_FIRST", 0.85),
        ),
        "ENGINE_PROD_GOD_RUBRIC1": getattr(
            engine_mod,
            "PROD_GOD_RUBRIC1",
            getattr(engine_mod, "GOD_PROBE_MIN_SECOND", 0.85),
        ),
        "ENGINE_TEST_EARLY_STOP": getattr(engine_mod, "TEST_EARLY_STOP", None),
        "ENGINE_LONG_MIN_OBJ_FOR_FIRST_OPEN": getattr(
            engine_mod, "LONG_MIN_OBJ_FOR_FIRST_OPEN", getattr(cfg, "LONG_MIN_OBJ_FOR_FIRST_OPEN", 8)
        ),
        "POLICY_PROD_GOD_ENABLE": getattr(policy_mod, "PROD_GOD_ENABLE", False),
        "POLICY_GOD_PROBE_MIN_FIRST": getattr(
            policy_mod, "GOD_PROBE_MIN_FIRST", 0.85
        ),
        "POLICY_GOD_PROBE_MIN_SECOND": getattr(
            policy_mod, "GOD_PROBE_MIN_SECOND", 0.85
        ),
        "POLICY_GOD_PROBE_DIFFICULTY": getattr(
            policy_mod, "GOD_PROBE_DIFFICULTY", 1.0
        ),
        "POLICY_GOD_PROBE_EXEMPT_FROM_CAP": getattr(
            policy_mod, "GOD_PROBE_EXEMPT_FROM_CAP", True
        ),
        "POLICY_PROD_GOD_RUBRIC0": getattr(
            policy_mod,
            "PROD_GOD_RUBRIC0",
            getattr(policy_mod, "GOD_PROBE_MIN_FIRST", 0.85),
        ),
        "POLICY_PROD_GOD_RUBRIC1": getattr(
            policy_mod,
            "PROD_GOD_RUBRIC1",
            getattr(policy_mod, "GOD_PROBE_MIN_SECOND", 0.85),
        ),
    }

    try:
        cfg.TEST_MODE = snapshots["TEST_MODE"]
        cfg.TEST_GOD_MIN_NORM = snapshots["TEST_GOD_MIN_NORM"]
        cfg.TEST_GOD_MAX_SE = snapshots["TEST_GOD_MAX_SE"]
        cfg.TEST_GOD_MIN_L2_SEEN = snapshots["TEST_GOD_MIN_L2_SEEN"]
        cfg.TEST_GOD_MIN_L2_ACC = snapshots["TEST_GOD_MIN_L2_ACC"]
        cfg.TEST_GOD_MIN_OPEN = snapshots["TEST_GOD_MIN_OPEN"]
        cfg.TEST_GOD_RUBRIC0 = snapshots["TEST_GOD_RUBRIC0"]
        cfg.TEST_GOD_RUBRIC1 = snapshots["TEST_GOD_RUBRIC1"]
        cfg.TEST_GOD_MIN_OPEN_RUBRIC = snapshots["TEST_GOD_MIN_OPEN_RUBRIC"]
        cfg.TEST_GOD_MIN_OPEN_RUBRIC_SEC = snapshots[
            "TEST_GOD_MIN_OPEN_RUBRIC_SEC"
        ]
        cfg.TEST_OPEN_FULL = snapshots["TEST_OPEN_FULL"]
        cfg.OPEN_RESERVE_FORCE = snapshots["OPEN_RESERVE_FORCE"]
        cfg.TEST_EARLY_STOP = snapshots["TEST_EARLY_STOP"]
        cfg.LONG_MIN_OBJ_FOR_FIRST_OPEN = snapshots[
            "LONG_MIN_OBJ_FOR_FIRST_OPEN"
        ]
        cfg.GOD_MIN_NORM = snapshots["GOD_MIN_NORM"]
        cfg.GOD_MAX_SE = snapshots["GOD_MAX_SE"]
        cfg.GOD_MIN_L2_SEEN = snapshots["GOD_MIN_L2_SEEN"]
        cfg.GOD_MIN_L2_ACC = snapshots["GOD_MIN_L2_ACC"]
        cfg.GOD_MIN_OPEN = snapshots["GOD_MIN_OPEN"]
        cfg.PROD_GOD_ENABLE = snapshots["PROD_GOD_ENABLE"]
        cfg.GOD_PROBE_MIN_FIRST = snapshots["GOD_PROBE_MIN_FIRST"]
        cfg.GOD_PROBE_MIN_SECOND = snapshots["GOD_PROBE_MIN_SECOND"]
        cfg.GOD_PROBE_DIFFICULTY = snapshots["GOD_PROBE_DIFFICULTY"]
        cfg.GOD_PROBE_EXEMPT_FROM_CAP = snapshots["GOD_PROBE_EXEMPT_FROM_CAP"]
        cfg.GOD_MIN_R1 = snapshots["GOD_MIN_R1"]
        cfg.GOD_MIN_R2 = snapshots["GOD_MIN_R2"]
        cfg.PROD_GOD_RUBRIC0 = snapshots["PROD_GOD_RUBRIC0"]
        cfg.PROD_GOD_RUBRIC1 = snapshots["PROD_GOD_RUBRIC1"]
        engine_mod.GOD_MIN_NORM = snapshots["ENGINE_GOD_MIN_NORM"]
        engine_mod.GOD_MAX_SE = snapshots["ENGINE_GOD_MAX_SE"]
        engine_mod.GOD_MIN_L2_SEEN = snapshots["ENGINE_GOD_MIN_L2_SEEN"]
        engine_mod.GOD_MIN_L2_ACC = snapshots["ENGINE_GOD_MIN_L2_ACC"]
        engine_mod.GOD_MIN_OPEN = snapshots["ENGINE_GOD_MIN_OPEN"]
        engine_mod.PROD_GOD_ENABLE = snapshots["ENGINE_PROD_GOD_ENABLE"]
        engine_mod.GOD_PROBE_MIN_FIRST = snapshots["ENGINE_GOD_PROBE_MIN_FIRST"]
        engine_mod.GOD_PROBE_MIN_SECOND = snapshots["ENGINE_GOD_PROBE_MIN_SECOND"]
        engine_mod.GOD_PROBE_DIFFICULTY = snapshots["ENGINE_GOD_PROBE_DIFFICULTY"]
        engine_mod.GOD_PROBE_EXEMPT_FROM_CAP = snapshots[
            "ENGINE_GOD_PROBE_EXEMPT_FROM_CAP"
        ]
        engine_mod.PROD_GOD_RUBRIC0 = snapshots["ENGINE_PROD_GOD_RUBRIC0"]
        engine_mod.PROD_GOD_RUBRIC1 = snapshots["ENGINE_PROD_GOD_RUBRIC1"]
        engine_mod.TEST_EARLY_STOP = snapshots["ENGINE_TEST_EARLY_STOP"]
        engine_mod.LONG_MIN_OBJ_FOR_FIRST_OPEN = snapshots[
            "ENGINE_LONG_MIN_OBJ_FOR_FIRST_OPEN"
        ]

        policy_mod.TEST_MODE = snapshots["POLICY_TEST_MODE"]
        policy_mod.TEST_OPEN_FULL = snapshots["POLICY_TEST_OPEN_FULL"]
        policy_mod.PROD_GOD_ENABLE = snapshots["POLICY_PROD_GOD_ENABLE"]
        policy_mod.GOD_PROBE_MIN_FIRST = snapshots["POLICY_GOD_PROBE_MIN_FIRST"]
        policy_mod.GOD_PROBE_MIN_SECOND = snapshots["POLICY_GOD_PROBE_MIN_SECOND"]
        policy_mod.GOD_PROBE_DIFFICULTY = snapshots["POLICY_GOD_PROBE_DIFFICULTY"]
        policy_mod.GOD_PROBE_EXEMPT_FROM_CAP = snapshots[
            "POLICY_GOD_PROBE_EXEMPT_FROM_CAP"
        ]
        policy_mod.PROD_GOD_RUBRIC0 = snapshots["POLICY_PROD_GOD_RUBRIC0"]
        policy_mod.PROD_GOD_RUBRIC1 = snapshots["POLICY_PROD_GOD_RUBRIC1"]
        policy_mod.TEST_EARLY_STOP = snapshots["POLICY_TEST_EARLY_STOP"]
        policy_mod.OPEN_RESERVE_FORCE = snapshots["POLICY_OPEN_RESERVE_FORCE"]

        if cfg.STAGING_PROFILE and _is_truthy(test_mode_flag):
            cfg.TEST_MODE = True
            policy_mod.TEST_MODE = True
            if test_min_norm is not None:
                value = float(test_min_norm)
                cfg.TEST_GOD_MIN_NORM = value
                cfg.GOD_MIN_NORM = value
                engine_mod.GOD_MIN_NORM = value
            if test_max_se is not None:
                value = float(test_max_se)
                cfg.TEST_GOD_MAX_SE = value
                cfg.GOD_MAX_SE = value
                engine_mod.GOD_MAX_SE = value
            if test_min_l2_seen is not None:
                value = int(test_min_l2_seen)
                cfg.TEST_GOD_MIN_L2_SEEN = value
                cfg.GOD_MIN_L2_SEEN = value
                engine_mod.GOD_MIN_L2_SEEN = value
            if test_min_l2_acc is not None:
                value = float(test_min_l2_acc)
                cfg.TEST_GOD_MIN_L2_ACC = value
                cfg.GOD_MIN_L2_ACC = value
                engine_mod.GOD_MIN_L2_ACC = value
            if test_min_open is not None:
                value = int(test_min_open)
                cfg.TEST_GOD_MIN_OPEN = value
                cfg.GOD_MIN_OPEN = value
                engine_mod.GOD_MIN_OPEN = value
            if test_rubric0 is not None:
                value = float(test_rubric0)
                cfg.TEST_GOD_RUBRIC0 = value
            if test_rubric1 is not None:
                value = float(test_rubric1)
                cfg.TEST_GOD_RUBRIC1 = value
            if test_rubric_primary is not None:
                cfg.TEST_GOD_MIN_OPEN_RUBRIC = float(test_rubric_primary)
            if test_rubric_secondary is not None:
                cfg.TEST_GOD_MIN_OPEN_RUBRIC_SEC = float(test_rubric_secondary)
            if test_open_full is not None:
                forced = _is_truthy(test_open_full)
                cfg.TEST_OPEN_FULL = forced
                policy_mod.TEST_OPEN_FULL = forced
            if getattr(cfg, "TEST_OPEN_FULL", False):
                cfg.OPEN_RESERVE_FORCE = True
                policy_mod.OPEN_RESERVE_FORCE = True
            if test_early_stop is not None:
                early_val = str(test_early_stop).strip().lower()
                if early_val in {"on", "off"}:
                    cfg.TEST_EARLY_STOP = early_val
                    engine_mod.TEST_EARLY_STOP = early_val
                    policy_mod.TEST_EARLY_STOP = early_val
                else:
                    cfg.TEST_EARLY_STOP = None
                    engine_mod.TEST_EARLY_STOP = None
                    policy_mod.TEST_EARLY_STOP = None
            if prod_god_enable is not None:
                enable_prod = _is_truthy(prod_god_enable)
                cfg.PROD_GOD_ENABLE = enable_prod
                engine_mod.PROD_GOD_ENABLE = enable_prod
                policy_mod.PROD_GOD_ENABLE = enable_prod

            engine_mod.GOD_PROBE_MIN_FIRST = cfg.GOD_PROBE_MIN_FIRST
            engine_mod.GOD_PROBE_MIN_SECOND = cfg.GOD_PROBE_MIN_SECOND
            engine_mod.GOD_PROBE_DIFFICULTY = cfg.GOD_PROBE_DIFFICULTY
            engine_mod.GOD_PROBE_EXEMPT_FROM_CAP = cfg.GOD_PROBE_EXEMPT_FROM_CAP
            cfg.PROD_GOD_RUBRIC0 = float(cfg.GOD_PROBE_MIN_FIRST)
            cfg.PROD_GOD_RUBRIC1 = float(cfg.GOD_PROBE_MIN_SECOND)
            engine_mod.PROD_GOD_RUBRIC0 = float(engine_mod.GOD_PROBE_MIN_FIRST)
            engine_mod.PROD_GOD_RUBRIC1 = float(engine_mod.GOD_PROBE_MIN_SECOND)
            policy_mod.GOD_PROBE_MIN_FIRST = cfg.GOD_PROBE_MIN_FIRST
            policy_mod.GOD_PROBE_MIN_SECOND = cfg.GOD_PROBE_MIN_SECOND
            policy_mod.GOD_PROBE_DIFFICULTY = cfg.GOD_PROBE_DIFFICULTY
            policy_mod.GOD_PROBE_EXEMPT_FROM_CAP = cfg.GOD_PROBE_EXEMPT_FROM_CAP
            policy_mod.PROD_GOD_RUBRIC0 = float(policy_mod.GOD_PROBE_MIN_FIRST)
            policy_mod.PROD_GOD_RUBRIC1 = float(policy_mod.GOD_PROBE_MIN_SECOND)

        prod_override_dict: Optional[Dict[str, object]] = None

        if prod_god_enable is not None:
            enable_prod = _is_truthy(prod_god_enable)
            cfg.PROD_GOD_ENABLE = enable_prod
            engine_mod.PROD_GOD_ENABLE = enable_prod
            policy_mod.PROD_GOD_ENABLE = enable_prod
        if long_min_open_floor is not None:
            value = int(long_min_open_floor)
            cfg.LONG_MIN_OBJ_FOR_FIRST_OPEN = value
            engine_mod.LONG_MIN_OBJ_FOR_FIRST_OPEN = value
            policy_mod.LONG_MIN_OBJ_FOR_FIRST_OPEN = value
        if long_min_obj_per_domain is not None:
            value = max(0, int(long_min_obj_per_domain))
            cfg.LONG_MIN_OBJ_PER_DOMAIN = value
            engine_mod.LONG_MIN_OBJ_PER_DOMAIN = value
            policy_mod.LONG_MIN_OBJ_PER_DOMAIN = value
            if value > 0:
                cfg.LONG_OBJ_MIN = value
                engine_mod.LONG_OBJ_MIN = value
                policy_mod.LONG_OBJ_MIN = value
                cfg.LONG_OBJ_MAX = value
                engine_mod.LONG_OBJ_MAX = value
                policy_mod.LONG_OBJ_MAX = value
        if long_estop_disable is not None:
            disable_flag = _is_truthy(long_estop_disable)
            enable_flag = not disable_flag
            cfg.LONG_EARLY_STOP_ENABLE = enable_flag
            engine_mod.LONG_EARLY_STOP_ENABLE = enable_flag
            policy_mod.LONG_EARLY_STOP_ENABLE = enable_flag
        if long_estop_min_obj is not None:
            value = max(1, int(long_estop_min_obj))
            cfg.LONG_OBJ_MIN = value
            engine_mod.LONG_OBJ_MIN = value
            policy_mod.LONG_OBJ_MIN = value
        if long_estop_conf is not None:
            value = float(long_estop_conf)
            cfg.LONG_EARLY_SE = value
            engine_mod.LONG_EARLY_SE = value
            policy_mod.LONG_EARLY_SE = value
        if test_early_stop is not None and not (cfg.STAGING_PROFILE and _is_truthy(test_mode_flag)):
            early_val = str(test_early_stop).strip().lower()
            if early_val in {"on", "off"}:
                cfg.TEST_EARLY_STOP = early_val
                engine_mod.TEST_EARLY_STOP = early_val
                policy_mod.TEST_EARLY_STOP = early_val
            else:
                cfg.TEST_EARLY_STOP = None
                engine_mod.TEST_EARLY_STOP = None
                policy_mod.TEST_EARLY_STOP = None
        enable_prod = bool(getattr(cfg, "PROD_GOD_ENABLE", False))
        if enable_prod:
            prod_override_dict = {
                "enabled": True,
                "min_norm": float(cfg.GOD_MIN_NORM),
                "max_se": float(cfg.GOD_MAX_SE),
                "min_l2_seen": int(cfg.GOD_MIN_L2_SEEN),
                "min_l2_acc": float(cfg.GOD_MIN_L2_ACC),
                "min_open": int(cfg.GOD_MIN_OPEN),
                "rubric0": float(getattr(cfg, "PROD_GOD_RUBRIC0", cfg.GOD_PROBE_MIN_FIRST)),
                "rubric1": float(getattr(cfg, "PROD_GOD_RUBRIC1", cfg.GOD_PROBE_MIN_SECOND)),
            }
            if prod_god_min_norm is not None:
                value = float(prod_god_min_norm)
                cfg.GOD_MIN_NORM = value
                engine_mod.GOD_MIN_NORM = value
                prod_override_dict["min_norm"] = value
            if prod_god_max_se is not None:
                value = float(prod_god_max_se)
                cfg.GOD_MAX_SE = value
                engine_mod.GOD_MAX_SE = value
                prod_override_dict["max_se"] = value
            if prod_god_min_l2_seen is not None:
                value = int(prod_god_min_l2_seen)
                cfg.GOD_MIN_L2_SEEN = value
                engine_mod.GOD_MIN_L2_SEEN = value
                prod_override_dict["min_l2_seen"] = value
            if prod_god_min_l2_acc is not None:
                value = float(prod_god_min_l2_acc)
                cfg.GOD_MIN_L2_ACC = value
                engine_mod.GOD_MIN_L2_ACC = value
                prod_override_dict["min_l2_acc"] = value
            if prod_god_min_open is not None:
                value = int(prod_god_min_open)
                cfg.GOD_MIN_OPEN = value
                engine_mod.GOD_MIN_OPEN = value
                prod_override_dict["min_open"] = value
            if prod_god_rubric0 is not None:
                value = float(prod_god_rubric0)
                cfg.GOD_MIN_R1 = value
                cfg.PROD_GOD_RUBRIC0 = value
                engine_mod.PROD_GOD_RUBRIC0 = value
                policy_mod.PROD_GOD_RUBRIC0 = value
                prod_override_dict["rubric0"] = value
            else:
                cfg.PROD_GOD_RUBRIC0 = float(getattr(cfg, "PROD_GOD_RUBRIC0", cfg.GOD_PROBE_MIN_FIRST))
            if prod_god_rubric1 is not None:
                value = float(prod_god_rubric1)
                cfg.GOD_MIN_R2 = value
                cfg.PROD_GOD_RUBRIC1 = value
                engine_mod.PROD_GOD_RUBRIC1 = value
                policy_mod.PROD_GOD_RUBRIC1 = value
                prod_override_dict["rubric1"] = value
            else:
                cfg.PROD_GOD_RUBRIC1 = float(getattr(cfg, "PROD_GOD_RUBRIC1", cfg.GOD_PROBE_MIN_SECOND))

        result = simulate(run, mode, seed)
        debug_meta: Optional[Dict[str, Any]] = None
        if cfg.STAGING_PROFILE:
            debug_meta = result.setdefault("debug", {})
            debug_meta["god_thresholds"] = cfg.god_thresholds()
            debug_meta.setdefault("open_floor", {})["min_obj"] = int(
                getattr(cfg, "LONG_MIN_OBJ_FOR_FIRST_OPEN", 0)
            )
            early_block = debug_meta.setdefault("early_stop", {})
            early_block["disable"] = not bool(getattr(cfg, "LONG_EARLY_STOP_ENABLE", True))
            early_block["min_obj"] = int(getattr(cfg, "LONG_OBJ_MIN", 0))
            early_block["conf"] = float(getattr(cfg, "LONG_EARLY_SE", 0.0))
            early_block["long_min_obj_per_domain"] = int(
                getattr(cfg, "LONG_MIN_OBJ_PER_DOMAIN", 0)
            )
        if prod_override_dict is not None:
            if debug_meta is None:
                debug_meta = result.setdefault("debug", {})
            echo = debug_meta.setdefault("god_thresholds_echo", {})
            echo["prod_overrides_applied"] = {
                **prod_override_dict,
                "probe_min_first": float(getattr(cfg, "GOD_PROBE_MIN_FIRST", 0.85)),
                "probe_min_second": float(getattr(cfg, "GOD_PROBE_MIN_SECOND", 0.85)),
                "probe_exempt": bool(getattr(cfg, "GOD_PROBE_EXEMPT_FROM_CAP", True)),
            }
        result["long_min_obj_per_domain"] = int(
            getattr(cfg, "LONG_MIN_OBJ_PER_DOMAIN", 0)
        )
        return result
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(400, str(exc)) from exc
    finally:
        cfg.TEST_MODE = snapshots["TEST_MODE"]
        cfg.TEST_GOD_MIN_NORM = snapshots["TEST_GOD_MIN_NORM"]
        cfg.TEST_GOD_MAX_SE = snapshots["TEST_GOD_MAX_SE"]
        cfg.TEST_GOD_MIN_L2_SEEN = snapshots["TEST_GOD_MIN_L2_SEEN"]
        cfg.TEST_GOD_MIN_L2_ACC = snapshots["TEST_GOD_MIN_L2_ACC"]
        cfg.TEST_GOD_MIN_OPEN = snapshots["TEST_GOD_MIN_OPEN"]
        cfg.TEST_GOD_RUBRIC0 = snapshots["TEST_GOD_RUBRIC0"]
        cfg.TEST_GOD_RUBRIC1 = snapshots["TEST_GOD_RUBRIC1"]
        cfg.TEST_GOD_MIN_OPEN_RUBRIC = snapshots["TEST_GOD_MIN_OPEN_RUBRIC"]
        cfg.TEST_GOD_MIN_OPEN_RUBRIC_SEC = snapshots[
            "TEST_GOD_MIN_OPEN_RUBRIC_SEC"
        ]
        cfg.TEST_OPEN_FULL = snapshots["TEST_OPEN_FULL"]
        cfg.OPEN_RESERVE_FORCE = snapshots["OPEN_RESERVE_FORCE"]
        policy_mod.TEST_MODE = snapshots["POLICY_TEST_MODE"]
        policy_mod.TEST_OPEN_FULL = snapshots["POLICY_TEST_OPEN_FULL"]
        policy_mod.OPEN_RESERVE_FORCE = snapshots["POLICY_OPEN_RESERVE_FORCE"]
        policy_mod.LONG_MIN_OBJ_FOR_FIRST_OPEN = snapshots[
            "POLICY_LONG_MIN_OBJ_FOR_FIRST_OPEN"
        ]
        cfg.LONG_MIN_OBJ_PER_DOMAIN = snapshots["LONG_MIN_OBJ_PER_DOMAIN"]
        engine_mod.LONG_MIN_OBJ_PER_DOMAIN = snapshots[
            "ENGINE_LONG_MIN_OBJ_PER_DOMAIN"
        ]
        policy_mod.LONG_MIN_OBJ_PER_DOMAIN = snapshots[
            "POLICY_LONG_MIN_OBJ_PER_DOMAIN"
        ]
        cfg.LONG_OBJ_MAX = snapshots["LONG_OBJ_MAX"]
        engine_mod.LONG_OBJ_MAX = snapshots["ENGINE_LONG_OBJ_MAX"]
        policy_mod.LONG_OBJ_MAX = snapshots["POLICY_LONG_OBJ_MAX"]
        cfg.LONG_OBJ_MIN = snapshots["LONG_OBJ_MIN"]
        engine_mod.LONG_OBJ_MIN = snapshots["ENGINE_LONG_OBJ_MIN"]
        policy_mod.LONG_OBJ_MIN = snapshots["POLICY_LONG_OBJ_MIN"]
        cfg.LONG_EARLY_STOP_ENABLE = snapshots["LONG_EARLY_STOP_ENABLE"]
        engine_mod.LONG_EARLY_STOP_ENABLE = snapshots[
            "ENGINE_LONG_EARLY_STOP_ENABLE"
        ]
        policy_mod.LONG_EARLY_STOP_ENABLE = snapshots[
            "POLICY_LONG_EARLY_STOP_ENABLE"
        ]
        cfg.LONG_EARLY_SE = snapshots["LONG_EARLY_SE"]
        engine_mod.LONG_EARLY_SE = snapshots["ENGINE_LONG_EARLY_SE"]
        policy_mod.LONG_EARLY_SE = snapshots["POLICY_LONG_EARLY_SE"]
        cfg.GOD_MIN_NORM = snapshots["GOD_MIN_NORM"]
        cfg.GOD_MAX_SE = snapshots["GOD_MAX_SE"]
        cfg.GOD_MIN_L2_SEEN = snapshots["GOD_MIN_L2_SEEN"]
        cfg.GOD_MIN_L2_ACC = snapshots["GOD_MIN_L2_ACC"]
        cfg.GOD_MIN_OPEN = snapshots["GOD_MIN_OPEN"]
        cfg.PROD_GOD_ENABLE = snapshots["PROD_GOD_ENABLE"]
        cfg.GOD_PROBE_MIN_FIRST = snapshots["GOD_PROBE_MIN_FIRST"]
        cfg.GOD_PROBE_MIN_SECOND = snapshots["GOD_PROBE_MIN_SECOND"]
        cfg.GOD_PROBE_DIFFICULTY = snapshots["GOD_PROBE_DIFFICULTY"]
        cfg.GOD_PROBE_EXEMPT_FROM_CAP = snapshots["GOD_PROBE_EXEMPT_FROM_CAP"]
        cfg.GOD_MIN_R1 = snapshots["GOD_MIN_R1"]
        cfg.GOD_MIN_R2 = snapshots["GOD_MIN_R2"]
        cfg.PROD_GOD_RUBRIC0 = snapshots["PROD_GOD_RUBRIC0"]
        cfg.PROD_GOD_RUBRIC1 = snapshots["PROD_GOD_RUBRIC1"]

        engine_mod.GOD_MIN_NORM = snapshots["ENGINE_GOD_MIN_NORM"]
        engine_mod.GOD_MAX_SE = snapshots["ENGINE_GOD_MAX_SE"]
        engine_mod.GOD_MIN_L2_SEEN = snapshots["ENGINE_GOD_MIN_L2_SEEN"]
        engine_mod.GOD_MIN_L2_ACC = snapshots["ENGINE_GOD_MIN_L2_ACC"]
        engine_mod.GOD_MIN_OPEN = snapshots["ENGINE_GOD_MIN_OPEN"]
        engine_mod.PROD_GOD_ENABLE = snapshots["ENGINE_PROD_GOD_ENABLE"]
        engine_mod.GOD_PROBE_MIN_FIRST = snapshots["ENGINE_GOD_PROBE_MIN_FIRST"]
        engine_mod.GOD_PROBE_MIN_SECOND = snapshots["ENGINE_GOD_PROBE_MIN_SECOND"]
        engine_mod.GOD_PROBE_DIFFICULTY = snapshots["ENGINE_GOD_PROBE_DIFFICULTY"]
        engine_mod.GOD_PROBE_EXEMPT_FROM_CAP = snapshots["ENGINE_GOD_PROBE_EXEMPT_FROM_CAP"]
        engine_mod.PROD_GOD_RUBRIC0 = snapshots["ENGINE_PROD_GOD_RUBRIC0"]
        engine_mod.PROD_GOD_RUBRIC1 = snapshots["ENGINE_PROD_GOD_RUBRIC1"]

        policy_mod.PROD_GOD_ENABLE = snapshots["POLICY_PROD_GOD_ENABLE"]
        policy_mod.GOD_PROBE_MIN_FIRST = snapshots["POLICY_GOD_PROBE_MIN_FIRST"]
        policy_mod.GOD_PROBE_MIN_SECOND = snapshots["POLICY_GOD_PROBE_MIN_SECOND"]
        policy_mod.GOD_PROBE_DIFFICULTY = snapshots["POLICY_GOD_PROBE_DIFFICULTY"]
        policy_mod.GOD_PROBE_EXEMPT_FROM_CAP = snapshots[
            "POLICY_GOD_PROBE_EXEMPT_FROM_CAP"
        ]
        policy_mod.PROD_GOD_RUBRIC0 = snapshots["POLICY_PROD_GOD_RUBRIC0"]
        policy_mod.PROD_GOD_RUBRIC1 = snapshots["POLICY_PROD_GOD_RUBRIC1"]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run headless adaptive simulation")
    parser.add_argument("--run", choices=["short", "long"], default="short")
    parser.add_argument("--mode", choices=["pass", "fail"], default="pass")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    data = simulate(args.run, args.mode, args.seed)
    print(json.dumps(data))


if __name__ == "__main__":  # pragma: no cover - CLI helper
    _main()


# Include router definitions after they have been declared so that the
# standalone FastAPI instance exposes identical routes to the staging wiring.
app.include_router(router)
