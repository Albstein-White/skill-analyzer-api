from __future__ import annotations
from typing import Tuple, Dict, Any
import re
from .llm_bridge import score_open

_DEFLECT_RX = re.compile(r"\b(i\s*don'?t\s*know|no\s*idea|pass|skip)\b", re.I)

def _clamp01(x: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return 0.0
    if xf < 0.0: return 0.0
    if xf > 1.0: return 1.0
    return xf

def _is_negative_sr(item) -> bool:
    pol = getattr(item, "polarity", None)
    if isinstance(pol, str) and pol.lower().startswith("neg"):
        return True
    try:
        iid = getattr(item, "id", None)
        return isinstance(iid, str) and iid.endswith("_neg")
    except Exception:
        return False

def _prompt_stub(item) -> str:
    for attr in ("prompt", "question", "stem", "text", "title"):
        v = getattr(item, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _score_mcq(item, value_idx: int) -> Tuple[float, Dict[str, Any]]:
    correct_idx = int(getattr(item, "correct", 0) or 0)
    is_correct = int(value_idx) == correct_idx
    credit = 1.0 if is_correct else 0.0
    meta = {"type": "MCQ", "correct_idx": correct_idx, "chosen_idx": int(value_idx)}
    return credit, meta

def _score_sjt(item, value_idx: int) -> Tuple[float, Dict[str, Any]]:
    """
    Supports weighted keys:
      - item.keys or item.sjt_keys = {idx: weight}
      - normalized to [0,1] across min..max
    Falls back to best/good/poor or exact-key if no weights.
    """
    keys = getattr(item, "keys", None) or getattr(item, "sjt_keys", None)
    if isinstance(keys, dict) and keys:
        # numeric indices only
        try:
            weights = {int(k): float(v) for k, v in keys.items()}
            wvals = list(weights.values())
            mn, mx = min(wvals), max(wvals)
            span = (mx - mn) if mx > mn else 1.0
            raw = float(weights.get(int(value_idx), mn))
            credit = (raw - mn) / span
            return _clamp01(credit), {"type": "SJT", "mode": "weights", "chosen_idx": int(value_idx)}
        except Exception:
            pass
    # fallback: best/good/poor indices if present
    for name in ("best_index", "best", "key_best"):
        if isinstance(getattr(item, name, None), int) and int(value_idx) == int(getattr(item, name)):
            return 1.0, {"type": "SJT", "mode": "best"}
    for name in ("good_index", "good", "key_good"):
        if isinstance(getattr(item, name, None), int) and int(value_idx) == int(getattr(item, name)):
            return 0.5, {"type": "SJT", "mode": "good"}
    for name in ("poor_index", "poor", "key_poor"):
        if isinstance(getattr(item, name, None), int) and int(value_idx) == int(getattr(item, name)):
            return 0.0, {"type": "SJT", "mode": "poor"}
    # last fallback: exact-key like MCQ
    return _score_mcq(item, value_idx)

def _score_sr(item, value_idx: int) -> Tuple[float, Dict[str, Any]]:
    neg = _is_negative_sr(item)
    v = int(value_idx)
    if v < 0: v = 0
    if v > 4: v = 4
    v = 4 - v if neg else v
    credit = _clamp01(v / 4.0)
    meta = {"type": "SR", "polarity_neg": bool(neg), "raw_idx": int(value_idx), "mapped_idx": int(v)}
    return credit, meta

def _score_open(item, text: str) -> Tuple[float, Dict[str, Any]]:
    ans = text if isinstance(text, str) else ""
    toks = len(re.findall(r"\w+", ans))
    if toks < 6 or _DEFLECT_RX.search(ans):
        return 0.0, {"type": "OPEN", "guard": "too_short_or_deflect", "tokens": toks}
    prompt = _prompt_stub(item)
    s = score_open(getattr(item, "id", "open"), ans, prompt_stub=prompt)
    credit = _clamp01(s)
    meta = {"type": "OPEN", "score": credit, "used_prompt_stub": bool(prompt)}
    return credit, meta

def score_item(item, answer) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (credit in 0..1, meta).
    MCQ/SJT/SR: answer.value is int index.
    OPEN: answer.value is str.
    """
    t = str(getattr(item, "type", "")).upper()
    val = getattr(answer, "value", None)
    if t == "MCQ":
        return _score_mcq(item, int(val))
    if t == "SJT":
        return _score_sjt(item, int(val))
    if t == "SR":
        return _score_sr(item, int(val))
    if t == "OPEN":
        return _score_open(item, str(val or ""))
    return 0.0, {"type": t or "UNKNOWN"}
