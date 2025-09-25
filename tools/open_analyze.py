from __future__ import annotations
import json
from pathlib import Path

def main():
    p = Path("llm_open_log.jsonl")
    if not p.exists(): 
        print("No llm_open_log.jsonl"); return
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip(): continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass
    if not rows:
        print("Log empty"); return
    # focus on latest run_id
    latest_run = rows[-1].get("run_id","")
    subset = [r for r in rows if r.get("run_id","")==latest_run] or rows
    last = {}
    for r in subset:
        if "item" in r:
            last[r["item"]] = r
    rows_html = []
    rows_html.append(f"<h2>OPEN audit (run: {latest_run})</h2>")
    rows_html.append("<table border='1' cellpadding='6' cellspacing='0'><tr><th>Item</th><th>Score</th><th>Prompt</th><th>Answer</th></tr>")
    for k in sorted(last):
        r = last[k]; sc = float(r.get("score", 0))
        prompt = (r.get("prompt","") or "").replace("<","&lt;")
        ans = (r.get("answer","") or "").replace("<","&lt;")
        rows_html.append(f"<tr><td>{k}</td><td>{sc:.2f}</td><td>{prompt}</td><td>{ans}</td></tr>")
    rows_html.append("</table>")
    out = Path("reports/open_audit.html"); out.write_text("\n".join(rows_html), encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
