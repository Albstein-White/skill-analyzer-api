from __future__ import annotations
from typing import Dict, Any, List

def _row(d: Dict[str, Any]) -> str:
    n = d.get('n', (d.get('asked_obj',0)+d.get('asked_open',0)))
    cap = d.get('cap', '')
    cap_mark = " *" if cap else ""
    return f"<tr><td>{d.get('domain')}{cap_mark}</td><td>{d.get('norm_score'):.1f}</td><td>{d.get('se'):.2f}</td><td>{n}</td></tr>"

def _row_breakdown(d: Dict[str, Any]) -> str:
    return f"<tr><td>{d.get('domain')}</td><td>{d.get('obj_pct','-')}</td><td>{d.get('open_pct','-')}</td><td>{d.get('sr_pct','-')}</td></tr>"

def export_report_html(result: Dict[str, Any], path: str) -> None:
    doms_raw = result.get("domain_scores", [])
    doms: List[Dict[str, Any]] = [dd if isinstance(dd, dict) else dd.__dict__ for dd in doms_raw]
    summ = result.get("summary", {}) or {}
    hidden = result.get("hidden_skills", []) or []
    overall = float(summ.get("mean", 0.0))
    rows = "\n".join(_row(d) for d in doms)
    rows2 = "\n".join(_row_breakdown(d) for d in doms)

    # Undervalued = hidden skills list intersects with domain rows
    uv = []
    if hidden:
        hdoms = [(getattr(h,'domain',None) or h.get('domain'), getattr(h,'confidence',None) or h.get('confidence'), getattr(h,'reason',None) or h.get('reason')) for h in hidden]
        dmap = {d['domain']: d for d in doms}
        uv_rows = []
        for dom, conf, reason in hdoms:
            dd = dmap.get(dom, {})
            uv_rows.append(f"<tr><td>{dom}</td><td>{dd.get('obj_pct','-')}</td><td>{dd.get('sr_pct','-')}</td><td>{reason}</td><td>{conf}</td></tr>")
        uv = ["<h3>Undervalued skills</h3>",
              "<table border='1' cellpadding='6' cellspacing='0'>",
              "<thead><tr><th>Domain</th><th>OBJ%</th><th>SR%</th><th>Gap</th><th>Confidence</th></tr></thead>",
              "<tbody>" + "\n".join(uv_rows) + "</tbody></table>"]

    foot = "<p><i>* capped due to no OPEN: ≤80.</i></p>"

    cats = f"""
    <h3>Performance categories</h3>
    <ul>
      <li><b>Speed</b>: {summ.get('speed_score',0):.1f}</li>
      <li><b>Precision</b>: {summ.get('precision_score',0):.1f}</li>
      <li><b>Consistency</b>: {summ.get('consistency_score',0):.1f}</li>
    </ul>
    """

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Skill Report</title>
<style>
 body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial}}
 .wrap{{max-width:960px;margin:40px auto;padding:0 16px}}
 h1{{margin:0 0 16px}}
 .overall{{font-size:1.1rem;margin:8px 0 16px}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{text-align:left}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Skill Report</h1>
  <div class="overall"><b>Overall:</b> {overall:.1f}</div>

  <table border='1' cellpadding='6' cellspacing='0'>
    <thead><tr><th>Domain</th><th>Score</th><th>SE</th><th>N</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {foot}

  <h3>Component breakdown (0–100)</h3>
  <table border='1' cellpadding='6' cellspacing='0'>
    <thead><tr><th>Domain</th><th>OBJ%</th><th>OPEN%</th><th>SR%</th></tr></thead>
    <tbody>{rows2}</tbody>
  </table>

  {''.join(uv) if uv else ''}

  {cats}

  <p><b>Timing:</b> SR {summ.get('avg_rt_sr',0)}s · MCQ {summ.get('avg_rt_mcq',0)}s · SJT {summ.get('avg_rt_sjt',0)}s · OPEN N={summ.get('oe_items',0)}</p>
</div>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
