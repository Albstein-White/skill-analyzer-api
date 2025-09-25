# tools/manual_cli.py
from __future__ import annotations
import argparse, json, os, sys, time
from typing import Any, List
from skill_core.engine import AdaptiveSession
from skill_core.types import Answer

def _text(it: Any) -> str:
    for k in ("prompt","stem","question","text","title","desc","description"):
        v = getattr(it, k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return f"{getattr(it,'id','')}"

def _opts(it: Any) -> List[str]:
    raw = getattr(it, "options", None) or getattr(it, "choices", None) or []
    out: List[str] = []
    for o in (raw or []):
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            for k in ("text","label","option","desc"):
                if isinstance(o.get(k), str) and o[k].strip():
                    out.append(o[k].strip()); break
            else:
                out.append(str(o))
        else:
            out.append(str(o))
    return out

def _ask_int(prompt: str, default: int = 0) -> int:
    try:
        s = input(prompt).strip()
        if s == "": return default
        return int(s)
    except Exception:
        return default

def ask_mcq(it):
    txt = _text(it); opts = _opts(it)
    print(f"\n[MCQ] {txt}")
    for i, opt in enumerate(opts): print(f"  {i}: {opt}")
    t0 = time.time(); idx = _ask_int("Choose index: ", 0); rt = time.time()-t0
    return Answer(item_id=it.id, value=idx, rt_sec=rt)

def ask_sjt(it):
    txt = _text(it); opts = _opts(it)
    print(f"\n[SJT] {txt}")
    for i, opt in enumerate(opts): print(f"  {i}: {opt}")
    t0 = time.time(); idx = _ask_int("Rate choice index: ", 0); rt = time.time()-t0
    return Answer(item_id=it.id, value=idx, rt_sec=rt)

def ask_sr(it):
    txt = _text(it)
    print(f"\n[SR] {txt}  (0..4; 0=lowest, 4=highest)")
    t0 = time.time(); idx = _ask_int("Choose 0..4: ", 2); rt = time.time()-t0
    return Answer(item_id=it.id, value=idx, rt_sec=rt)

def ask_open(it):
    txt = _text(it)
    print(f"\n[OPEN] {txt}")
    t0 = time.time(); ans = input("Your answer: ").strip(); rt = time.time()-t0
    return Answer(item_id=it.id, value=ans, rt_sec=rt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["short","long"], default="short")
    ap.add_argument("--llm", choices=["none","azure","ollama"], default="none")
    a = ap.parse_args()
    if a.llm != "none":
        os.environ["USE_LLM_OPEN"] = "1"; os.environ["LLM_BACKEND"] = a.llm

    sess = AdaptiveSession(run_type=a.run)
    print(f"Manual {a.run} test. Ctrl+C to exit.")
    try:
        while True:
            it = sess.next_item()
            if it is None: break
            t = str(getattr(it, "type", "MCQ")).upper()
            print(f"\n--- {t} | {getattr(it,'domain','')} | id={getattr(it,'id','')} ---")
            if t == "MCQ":   ans = ask_mcq(it)
            elif t == "SJT": ans = ask_sjt(it)
            elif t == "SR":  ans = ask_sr(it)
            elif t == "OPEN":ans = ask_open(it)
            else:
                ans = Answer(item_id=it.id, value=0, rt_sec=0.0)
            sess.answer_current(ans)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    res = sess.finalize()
    d = json.loads(json.dumps(res, default=lambda o: getattr(o, "__dict__", o)))
    from skill_core.report_html import export_report_html
    os.makedirs("reports", exist_ok=True)
    out = f"reports/manual_{a.run}.html"
    export_report_html(d, out)
    print(f"Report: {out}")

if __name__ == "__main__":
    main()
