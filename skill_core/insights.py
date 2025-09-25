# skill_core/insights.py
from __future__ import annotations
import json, os
from statistics import mean
from typing import Dict, List, Tuple

def _load_calibration(path: str = "calibration.json") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"slack": 0.10, "backends": {"default": {"floor": 0.05, "target": 0.85}}}

def _clamp_open(x: float, backend: str, calib: dict) -> float:
    b = calib.get("backends", {}).get(backend, calib.get("backends", {}).get("default", {}))
    floor = float(b.get("floor", 0.05))
    target = float(b.get("target", 0.85))
    # map to [floor, target] by simple clamp
    return max(floor, min(target, x))

def _sr_norm_from_likert(vals: List[int]) -> float:
    # Likert 1..5 → 0..1
    if not vals: return 0.5
    return mean([(v - 1) / 4.0 for v in vals])

def compute_undervalued(
    backend: str,
    domain_rows: Dict[str, dict],
    sr_raw: Dict[str, List[Tuple[str,int,bool]]],
    slack: float | None = None,
    cap: int = 5,
) -> List[Dict[str, float]]:
    """
    domain_rows: { domain: {"obj": float(0..1), "se": float, ...} }  (use your computed objective mean & SE)
    sr_raw: { domain: [ (item_id, likert_1_to_5, is_negative), ... ] }
    """
    calib = _load_calibration()
    slack_val = slack if slack is not None else float(calib.get("slack", 0.10))
    out = []
    for dom, row in domain_rows.items():
        obj = float(row.get("obj", 0.0))
        se  = float(row.get("se", 0.5))
        # apply backend clamp for OPEN already folded into obj:
        obj_adj = _clamp_open(obj, backend, calib)

        # build SR series (mirror negatives)
        sr_vals = []
        for _, v, neg in sr_raw.get(dom, []):
            v = int(v)
            v = 6 - v if neg else v      # mirror *_neg
            sr_vals.append(v)
        sr = _sr_norm_from_likert(sr_vals)

        delta = obj_adj - sr
        if sr <= 0.30 and obj_adj >= 0.70 and delta >= max(0.25, slack_val) and se <= 0.40:
            out.append({"domain": dom, "delta": round(delta, 3), "sr": round(sr,3), "obj": round(obj_adj,3), "se": round(se,3)})

    out.sort(key=lambda d: d["delta"], reverse=True)
    return out[:cap]
