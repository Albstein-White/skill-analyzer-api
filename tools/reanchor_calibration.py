# tools/reanchor_calibration.py
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from skill_core.engine import AdaptiveSession  # noqa
from skill_core.calibration import _to_dict  # reuse helper

def _answer_for(item, profile: str):
    from autoplay import _sjt_index, _pick_definitely_wrong_mcq, _open_text_for  # type: ignore
    from skill_core.types import Answer
    t = str(getattr(item, "type", "MCQ")).upper()
    iid = getattr(item, "id", "")
    if profile == "all-wrong":
        if t == "MCQ":  return Answer(item_id=iid, value=_pick_definitely_wrong_mcq(item), rt_sec=0.9)
        if t == "SJT":  return Answer(item_id=iid, value=_sjt_index(item, "poor"), rt_sec=1.1)
        if t == "SR":   return Answer(item_id=iid, value=0, rt_sec=1.0)
        if t == "OPEN": return Answer(item_id=iid, value="I don't know.", rt_sec=2.0)
    if profile == "perfect":
        if t == "MCQ":  return Answer(item_id=iid, value=int(getattr(item,"correct",0) or 0), rt_sec=0.8)
        if t == "SJT":  return Answer(item_id=iid, value=_sjt_index(item, "best"), rt_sec=1.0)
        if t == "SR":   return Answer(item_id=iid, value=4, rt_sec=0.9)
        if t == "OPEN": return Answer(item_id=iid, value=_open_text_for(item), rt_sec=3.0)
    if t == "MCQ":  return Answer(item_id=iid, value=_pick_definitely_wrong_mcq(item), rt_sec=1.0)
    if t == "SJT":  return Answer(item_id=iid, value=_sjt_index(item, "good"), rt_sec=1.2)
    if t == "SR":   return Answer(item_id=iid, value=2, rt_sec=1.1)
    return Answer(item_id=iid, value="pass", rt_sec=2.0)

def _scores(run_type: str, profile: str) -> dict[str,float]:
    sess = AdaptiveSession(run_type=run_type)
    n = 0
    while True:
        it = sess.next_item()
        if it is None: break
        sess.answer_current(_answer_for(it, profile))
        n += 1
    if n == 0: raise RuntimeError("0 items answered during reanchor.")
    res = _to_dict(sess.finalize())
    return {d["domain"]: float(d["norm_score"]) for d in res.get("domain_scores", [])}

def main():
    best = _scores("long", "perfect")
    worst = _scores("long", "all-wrong")
    out = {"slack": 0.10, "min_span": 40.0, "floor": 10.0, "ceil": 100.0, "domains": {}}
    for k in sorted(set(best) | set(worst)):
        b = float(best.get(k, 60.0))
        w = float(worst.get(k, 40.0))
        # enforce correct ordering and some separation
        if b <= w: b, w = w + 1.0, w  # force best>worst by at least 1
        out["domains"][k] = {"worst": round(w, 1), "best": round(b, 1)}
    path = ROOT / "skill_core" / "calibration_v2.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path}")

if __name__ == "__main__":
    main()
