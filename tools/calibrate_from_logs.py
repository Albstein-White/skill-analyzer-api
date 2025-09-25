# tools/calibrate_from_logs.py
import json, io, os, glob, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "skill_core" / "data"
DATA.mkdir(parents=True, exist_ok=True)
CAL_PATH = DATA / "calibration.json"

def _safe_load_lines(p):
    out=[]
    try:
        with io.open(p,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: out.append(json.loads(line))
                except: pass
    except FileNotFoundError:
        pass
    return out

def _scan_reports(pattern):
    vals=[]
    for p in sorted(glob.glob(pattern)):
        try:
            with io.open(p,"r",encoding="utf-8") as f:
                t=f.read()
            # naive scrape for composite lines: "Composite: 0.83 (OPEN=..., OBJ=...)"
            for ln in t.splitlines():
                if "Composite:" in ln:
                    try:
                        x=float(ln.split("Composite:")[1].split()[0])
                        vals.append(x)
                    except: pass
        except: pass
    return vals

def main():
    # 1) Collect composites from recent HTML reports (fallback if JSONL missing)
    reps = []
    reps += _scan_reports(str(ROOT/"reports"/"auto_long_perfect_20250923_171344.html"))
    reps += _scan_reports(str(ROOT/"reports"/"auto_long_all-wrong_20250923_171430.html"))
    reps += _scan_reports(str(ROOT/"reports"/"auto_long_sr_low_obj_high_20250923_190021.html"))

    # 2) Optionally mine llm_open_log.jsonl for backend info (best-effort)
    jpath = ROOT / "llm_open_log.jsonl"
    jlines = _safe_load_lines(jpath)
    backends = set()
    for r in jlines:
        be = r.get("backend") or r.get("meta",{}).get("backend")
        if be: backends.add(be)
    if not backends:
        # fall back to known key names used in your code
        backends = {"azure"}  # adjust if needed

    # Heuristic anchors:
    # floor = min(reported composites) or 0.05
    # target = max(reported composites) or 0.85
    C_floor = min(reps) if reps else 0.05
    C_target = max(reps) if reps else 0.85

    # guardrails
    C_floor = max(0.0, min(0.20, C_floor))
    C_target = max(0.60, min(0.99, C_target))

    cal = {"slack": 0.10, "epsilon": 0.01, "backends": {}}
    for be in backends:
        cal["backends"][be] = {"floor": C_floor, "target": C_target}

    with io.open(CAL_PATH,"w",encoding="utf-8") as f:
        json.dump(cal,f,indent=2)
    print(f"Wrote {CAL_PATH}")
    print(f"Backends: {', '.join(backends)}  floor={C_floor:.3f}  target={C_target:.3f}")

if __name__=="__main__":
    main()
