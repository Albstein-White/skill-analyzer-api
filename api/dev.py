from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, Optional, Sequence

from fastapi import APIRouter, HTTPException, Query

import skill_core.engine as engine_mod
from skill_core.scoring import score_item
from skill_core.types import Answer
from skill_core import config as cfg
import skill_core.policy as policy_mod
from skill_core.engine import AdaptiveSession

router = APIRouter()


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
        "tiers": tiers,
        "reasons": reasons,
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
    test_open_full: Optional[str] = Query(None, alias="TEST_OPEN_FULL"),
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
        "TEST_OPEN_FULL": getattr(cfg, "TEST_OPEN_FULL", False),
        "OPEN_RESERVE_FORCE": getattr(cfg, "OPEN_RESERVE_FORCE", False),
        "POLICY_TEST_MODE": getattr(policy_mod, "TEST_MODE", False),
        "POLICY_TEST_OPEN_FULL": getattr(policy_mod, "TEST_OPEN_FULL", False),
        "POLICY_OPEN_RESERVE_FORCE": getattr(policy_mod, "OPEN_RESERVE_FORCE", False),
        "GOD_MIN_NORM": getattr(cfg, "GOD_MIN_NORM", 0.0),
        "GOD_MAX_SE": getattr(cfg, "GOD_MAX_SE", 0.0),
        "GOD_MIN_L2_SEEN": getattr(cfg, "GOD_MIN_L2_SEEN", 0),
        "GOD_MIN_L2_ACC": getattr(cfg, "GOD_MIN_L2_ACC", 0.0),
        "GOD_MIN_OPEN": getattr(cfg, "GOD_MIN_OPEN", 0),
        "ENGINE_GOD_MIN_NORM": getattr(engine_mod, "GOD_MIN_NORM", 0.0),
        "ENGINE_GOD_MAX_SE": getattr(engine_mod, "GOD_MAX_SE", 0.0),
        "ENGINE_GOD_MIN_L2_SEEN": getattr(engine_mod, "GOD_MIN_L2_SEEN", 0),
        "ENGINE_GOD_MIN_L2_ACC": getattr(engine_mod, "GOD_MIN_L2_ACC", 0.0),
        "ENGINE_GOD_MIN_OPEN": getattr(engine_mod, "GOD_MIN_OPEN", 0),
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
        cfg.TEST_OPEN_FULL = snapshots["TEST_OPEN_FULL"]
        cfg.OPEN_RESERVE_FORCE = snapshots["OPEN_RESERVE_FORCE"]
        cfg.GOD_MIN_NORM = snapshots["GOD_MIN_NORM"]
        cfg.GOD_MAX_SE = snapshots["GOD_MAX_SE"]
        cfg.GOD_MIN_L2_SEEN = snapshots["GOD_MIN_L2_SEEN"]
        cfg.GOD_MIN_L2_ACC = snapshots["GOD_MIN_L2_ACC"]
        cfg.GOD_MIN_OPEN = snapshots["GOD_MIN_OPEN"]
        engine_mod.GOD_MIN_NORM = snapshots["ENGINE_GOD_MIN_NORM"]
        engine_mod.GOD_MAX_SE = snapshots["ENGINE_GOD_MAX_SE"]
        engine_mod.GOD_MIN_L2_SEEN = snapshots["ENGINE_GOD_MIN_L2_SEEN"]
        engine_mod.GOD_MIN_L2_ACC = snapshots["ENGINE_GOD_MIN_L2_ACC"]
        engine_mod.GOD_MIN_OPEN = snapshots["ENGINE_GOD_MIN_OPEN"]

        policy_mod.TEST_MODE = snapshots["POLICY_TEST_MODE"]
        policy_mod.TEST_OPEN_FULL = snapshots["POLICY_TEST_OPEN_FULL"]
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
            if test_open_full is not None:
                forced = _is_truthy(test_open_full)
                cfg.TEST_OPEN_FULL = forced
                policy_mod.TEST_OPEN_FULL = forced
            if getattr(cfg, "TEST_OPEN_FULL", False):
                cfg.OPEN_RESERVE_FORCE = True
                policy_mod.OPEN_RESERVE_FORCE = True

        return simulate(run, mode, seed)
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
        cfg.TEST_OPEN_FULL = snapshots["TEST_OPEN_FULL"]
        cfg.OPEN_RESERVE_FORCE = snapshots["OPEN_RESERVE_FORCE"]
        policy_mod.TEST_MODE = snapshots["POLICY_TEST_MODE"]
        policy_mod.TEST_OPEN_FULL = snapshots["POLICY_TEST_OPEN_FULL"]
        policy_mod.OPEN_RESERVE_FORCE = snapshots["POLICY_OPEN_RESERVE_FORCE"]
        cfg.GOD_MIN_NORM = snapshots["GOD_MIN_NORM"]
        cfg.GOD_MAX_SE = snapshots["GOD_MAX_SE"]
        cfg.GOD_MIN_L2_SEEN = snapshots["GOD_MIN_L2_SEEN"]
        cfg.GOD_MIN_L2_ACC = snapshots["GOD_MIN_L2_ACC"]
        cfg.GOD_MIN_OPEN = snapshots["GOD_MIN_OPEN"]
        engine_mod.GOD_MIN_NORM = snapshots["ENGINE_GOD_MIN_NORM"]
        engine_mod.GOD_MAX_SE = snapshots["ENGINE_GOD_MAX_SE"]
        engine_mod.GOD_MIN_L2_SEEN = snapshots["ENGINE_GOD_MIN_L2_SEEN"]
        engine_mod.GOD_MIN_L2_ACC = snapshots["ENGINE_GOD_MIN_L2_ACC"]
        engine_mod.GOD_MIN_OPEN = snapshots["ENGINE_GOD_MIN_OPEN"]


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
