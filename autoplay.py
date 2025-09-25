# autoplay.py
from __future__ import annotations
import argparse, os, json, random, datetime
from typing import Any, Dict, Optional
from skill_core.question_bank import load_bank
from skill_core.report_html import export_report_html
from skill_core.types import Answer as A

def _new_run_id() -> str:
    return datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

# per-domain OPEN texts
OPEN_BY_DOMAIN: Dict[str, str] = {
    "Analytical": ("Scan histogram/boxplot; compute z-score and IQR fences; flag |z|>3 or outside [Q1−1.5·IQR, Q3+1.5·IQR]. "
                   "Diagnose cause; compare model with/without using MAE/RMSE and CV; remove only if error drops and residuals stabilize."),
    "Mathematical": ("Plan 14 days: D1–2 rules; D3–6 60 mixed/day; D7 error-log drill; D8–12 80 timed/day; D13 mock; D14 review. "
                     "Track accuracy %, sec/problem, and error categories; raise difficulty on ≥90% and ≤20s/problem."),
    "Verbal": ("One-sentence summary template: {Subject} did {main action} because {reason}, leading to {result}. "
               "Checks: (1) Is the cause explicit? (2) Can a peer answer “so what?” from the sentence alone?"),
    "Memory": ("Make 30 Q/A cards. Daily 15-min recall. Spacing D1, D2, D3, D5, D7. "
               "Rule: correct→+2 days; wrong→repeat after 10 min and next day. Metric: 48h and 7d delayed recall %."),
    "Spatial": ("Pick a scale, draw envelope, place anchors, rotate candidate shape by 90° increments; "
                "verify bounding-box fits, clearances ≥75 cm, and door swings. Document decision rule and failure cases."),
    "Creativity": ("Three paperclip classroom uses: conductivity probe; pop-up hinge; mini armature. "
                   "Metric: novelty votes from 5 peers (≥60% unseen) and constraint fit."),
    "Strategy": ("MoSCoW scope; cut could/won’t; vertical walking skeleton; freeze interfaces; daily risk review. "
                 "Metric: burn-up to MVP and critical-path slack."),
    "Social": ("1: Private 1:1 with each; 2: joint goals; 3: agree behaviors and check-ins; 4: document. "
               "Metric: joint deliverable on time and post-meeting pulse ≥4/5 both sides."),
}

def _open_text_for(item: Any) -> str:
    dom = getattr(item, "domain", "") or ""
    txt = OPEN_BY_DOMAIN.get(dom)
    if isinstance(txt, str) and txt.strip():
        return txt
    return "State goal, method, metric. Prioritize by impact/effort, validate quickly, and report a single success KPI."

def _mcq_correct_idx(it) -> int:
    corr = getattr(it, "correct", None)
    opts = getattr(it, "options", None)
    if isinstance(corr, int) and isinstance(opts, (list, tuple)):
        if 0 <= corr < len(opts): return corr
        if 1 <= corr <= len(opts): return corr - 1
    if isinstance(corr, str):
        try:
            idx = int(corr)
            if isinstance(opts, (list, tuple)):
                if 0 <= idx < len(opts): return idx
                if 1 <= idx <= len(opts): return idx - 1
        except Exception:
            pass
        if isinstance(opts, (list, tuple)):
            for i, o in enumerate(opts):
                if o == corr: return i
    return 0

def _pick_definitely_wrong_mcq(it) -> int:
    ci = _mcq_correct_idx(it)
    opts = getattr(it, "options", None)
    if isinstance(opts, (list, tuple)) and len(opts) >= 2:
        return (ci + 1) % len(opts)
    return 0

def _sjt_index(it, which: str) -> int:
    keys = getattr(it, "keys", None) or getattr(it, "sjt_keys", None)
    if isinstance(keys, dict) and keys:
        items = sorted(((int(k), float(v)) for k, v in keys.items()), key=lambda kv: kv[1], reverse=True)
        if which == "best": return int(items[0][0])
        if which == "poor": return int(items[-1][0])
        return int(items[1][0] if len(items) > 1 else items[0][0])
    for name in (f"{which}_index", which, f"key_{which}", f"{which}_key"):
        v = getattr(it, name, None)
        if isinstance(v, int): return v
    n = len(getattr(it, "options", []) or [])
    ci = _mcq_correct_idx(it)
    if which == "best": return ci
    if which == "poor": return (ci + 1) % max(1, n or 4)
    return 1 if n > 2 else (ci + 1) % max(1, n or 4)

def _is_neg_sr(it) -> bool:
    pol = getattr(it, "polarity", None)
    if isinstance(pol, str) and pol.lower().startswith("neg"): return True
    iid = getattr(it, "id", "")
    return isinstance(iid, str) and iid.endswith("_neg")

def _answer_for(item: Any, profile: str) -> A:
    t = str(getattr(item, "type", "MCQ")).upper()
    iid = getattr(item, "id", "")
    if profile == "all-wrong":
        if t == "MCQ":  return A(item_id=iid, value=_pick_definitely_wrong_mcq(item), rt_sec=0.9)
        if t == "SJT":  return A(item_id=iid, value=_sjt_index(item, "poor"), rt_sec=1.1)
        if t == "SR":   return A(item_id=iid, value=(4 if _is_neg_sr(item) else 0), rt_sec=1.0)
        if t == "OPEN": return A(item_id=iid, value="I don't know.", rt_sec=2.0)
    if profile == "perfect":
        if t == "MCQ":  return A(item_id=iid, value=_mcq_correct_idx(item), rt_sec=0.8)
        if t == "SJT":  return A(item_id=iid, value=_sjt_index(item, "best"), rt_sec=1.0)
        if t == "SR":   return A(item_id=iid, value=(0 if _is_neg_sr(item) else 4), rt_sec=0.9)
        if t == "OPEN": return A(item_id=iid, value=_open_text_for(item), rt_sec=3.0)
    if t == "MCQ":  return A(item_id=iid, value=_pick_definitely_wrong_mcq(item), rt_sec=1.0)
    if t == "SJT":  return A(item_id=iid, value=_sjt_index(item, "good"), rt_sec=1.2)
    if t == "SR":   return A(item_id=iid, value=2, rt_sec=1.1)
    return A(item_id=iid, value="pass", rt_sec=2.0)

def run(run_type: str, profile: str, seed: Optional[int], backend: str):
    random.seed(seed or 1234); _ = load_bank()
    from skill_core.engine import AdaptiveSession
    sess = AdaptiveSession(run_type=("long" if run_type == "long" else "short"))

    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    os.environ["RUN_ID"] = run_id; os.environ["PROFILE"] = profile
    if os.getenv("CLEAR_OPEN_LOG", "1") == "1":
        try: open("llm_open_log.jsonl", "w", encoding="utf-8").close()
        except Exception: pass

    answered = 0
    while True:
        it = sess.next_item()
        if it is None: break
        sess.answer_current(_answer_for(it, profile)); answered += 1
    if answered <= 0: raise RuntimeError("Driver answered 0 items.")

    res = sess.finalize()
    if not isinstance(res, dict):
        res = json.loads(json.dumps(res, default=lambda o: getattr(o, "__dict__", o)))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    base = f"auto_{run_type}_{profile}_{ts}"
    export_report_html(res, os.path.join("reports", base + ".html"))
    print(f"Report: reports\\{base}.html")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["short", "long"], default="long")
    ap.add_argument("--profile", choices=["perfect", "all-wrong", "none"], default="perfect")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--llm", choices=["none", "ollama", "azure"], default="none")
    a = ap.parse_args()
    if a.llm != "none":
        os.environ["USE_LLM_OPEN"] = "1"; os.environ["LLM_BACKEND"] = a.llm
    run(a.run, a.profile, a.seed, backend=a.llm)

if __name__ == "__main__":
    main()
