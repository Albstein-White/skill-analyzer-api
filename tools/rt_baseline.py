from __future__ import annotations
import csv, json
from pathlib import Path
from statistics import median

def main():
    rows = []
    for p in Path("reports").glob("*.items.csv"):
        with p.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    t = row["type"].upper()
                    rt = float(row["rt_sec"])
                    if t in ("MCQ","SJT","SR","OPEN") and rt > 0:
                        rows.append((t, rt))
                except Exception:
                    pass
    if not rows:
        print("No item CSVs found.")
        return
    by = {"MCQ": [], "SJT": [], "SR": [], "OPEN": []}
    for t, rt in rows:
        by[t].append(rt)
    base = {}
    for t in by:
        vals = sorted(by[t])
        if not vals: continue
        # between median and P80 for robustness
        p50 = median(vals)
        p80 = vals[min(int(0.8*len(vals)), len(vals)-1)]
        base[t] = round((p50 + p80) / 2.0, 2)
    if not base:
        print("No baselines computed.")
        return
    Path(".").joinpath("rt_baseline.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    print("Wrote rt_baseline.json:", base)

if __name__ == "__main__":
    main()
