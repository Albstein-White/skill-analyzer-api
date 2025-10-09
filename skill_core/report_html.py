from __future__ import annotations
from typing import Dict, Any, List

from .config import AUDIT_EXPORT_ENABLED

def _row(d: Dict[str, Any]) -> str:
    n = d.get('n', (d.get('asked_obj',0)+d.get('asked_open',0)))
    cap = d.get('cap', '')
    cap_mark = " *" if cap else ""
    return f"<tr><td>{d.get('domain')}{cap_mark}</td><td>{d.get('norm_score'):.1f}</td><td>{d.get('se'):.2f}</td><td>{n}</td></tr>"

def _row_breakdown(d: Dict[str, Any]) -> str:
    return f"<tr><td>{d.get('domain')}</td><td>{d.get('obj_pct','-')}</td><td>{d.get('open_pct','-')}</td></tr>"

def export_report_html(result: Dict[str, Any], path: str) -> None:
    doms_raw = result.get("domain_scores", [])
    doms: List[Dict[str, Any]] = [dd if isinstance(dd, dict) else dd.__dict__ for dd in doms_raw]
    summ = result.get("summary", {}) or {}
    meta = result.get("meta", {}) or {}
    hidden = result.get("hidden_skills", []) or []
    overall = float(summ.get("mean", 0.0))
    rows = "\n".join(_row(d) for d in doms)
    rows2 = "\n".join(_row_breakdown(d) for d in doms)

    run_type = str(result.get("run_type") or (result.get("meta") or {}).get("run") or "").lower()

    incomplete_html = ""
    if meta.get("incomplete") and meta.get("incomplete_reason") == "CAP_BEFORE_MINIMA":
        shortfalls = meta.get("shortfalls") or {}
        if isinstance(shortfalls, dict) and shortfalls:
            items: List[str] = []
            for dom, val in shortfalls.items():
                try:
                    needed = int(val)
                except Exception:
                    needed = val
                items.append(f"<li>{dom}: needs {needed} more objective item(s)</li>")
            if items:
                incomplete_html = (
                    "<div class=\"banner warning\">"
                    "Run hit step cap before completing minimum objectives"
                    f"<ul>{''.join(items)}</ul>"
                    "</div>"
                )

    plan_entries = result.get("plan") or []
    plan_section = ""
    if plan_entries and run_type == "long":
        blocks: List[str] = []
        for entry in plan_entries:
            if not isinstance(entry, dict):
                continue
            goals = entry.get("goals") or []
            items: List[str] = []
            for goal in goals:
                if not isinstance(goal, dict):
                    continue
                g_type = str(goal.get("type") or "Goal").capitalize()
                text = goal.get("text") or ""
                measure = goal.get("measure")
                tag = goal.get("tag")
                extras = []
                if tag:
                    extras.append(f"focus: {tag}")
                if measure:
                    extras.append(str(measure))
                extra_txt = f" <span>({' · '.join(extras)})</span>" if extras else ""
                items.append(f"<li><b>{g_type}</b>: {text}{extra_txt}</li>")
            lvl = entry.get("level") or "0"
            blocks.append(
                '<div>'
                + f"<h4>{entry.get('domain')} (level {lvl})</h4>"
                + f"<ul>{''.join(items)}</ul>"
                + '</div>'
            )
        if blocks:
            plan_section = '<h3>Improvement Plan</h3>' + ''.join(blocks)

    open_sections: List[str] = []
    for d in doms:
        contribs = d.get('open_contrib') or []
        if not contribs:
            continue
        lines: List[str] = []
        for entry in contribs:
            try:
                b = int(entry.get('b', 0))
            except Exception:
                b = 0
            r = float(entry.get('r', 0.0) or 0.0)
            mu = float(entry.get('mu', 0.0) or 0.0)
            dtheta = float(entry.get('dtheta', 0.0) or 0.0)
            scaled = ' (capped)' if entry.get('scaled') else ''
            ignored = entry.get('ignored')
            note = f" (ignored: {ignored})" if ignored else ''
            lines.append(
                f"<li>OPEN @ {b:+d}: r={r:.2f} vs expected mu={mu:.2f} -> delta_theta={dtheta:+.3f}{scaled}{note}</li>"
            )
        theta_val = float(d.get('theta', 0.0) or 0.0)
        se_val = float(d.get('se', 0.0) or 0.0)
        open_sections.append(
            '<div>'
            + f"<h4>{d.get('domain')}</h4>"
            + f"<p>OPEN count: {int(d.get('open_count', 0) or 0)} | theta={theta_val:.3f} | SE={se_val:.3f}</p>"
            + f"<ul>{''.join(lines)}</ul>"
            + '</div>'
        )

    open_summary = ''
    if open_sections:
        open_summary = '<h3>OPEN contributions</h3>' + ''.join(open_sections)

    reliability_sections: List[str] = []
    reliability_flagged = False
    for d in doms:
        rel = d.get('reliability') or {}
        mirror_ok = bool(rel.get('sr_mirror_ok', True))
        trap_count = int(rel.get('trap_count', 0) or 0)
        mirror_txt = 'OK' if mirror_ok else 'Needs review'
        reliability_sections.append(
            '<div>'
            + f"<h4>{d.get('domain')}</h4>"
            + f"<p>Mirror consistency: {mirror_txt} · Trap flags: {trap_count}</p>"
            + '</div>'
        )
        if not mirror_ok or trap_count > 0:
            reliability_flagged = True

    reliability_html = ''
    if reliability_sections:
        reliability_html = '<h3>Self-report &amp; reliability</h3>' + ''.join(reliability_sections)

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

    audit_links = ""
    if AUDIT_EXPORT_ENABLED:
        report_id = result.get("reportId") or meta.get("reportId")
        if report_id:
            rid = str(report_id)
            audit_links = (
                "<p class=\"audit-links\">"
                f"<a href=\"/results/{rid}/audit.json\">Download audit (JSON)</a> · "
                f"<a href=\"/results/{rid}/audit.csv\">Download audit (CSV)</a>"
                "</p>"
            )

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
 .banner{{padding:12px 16px;border-radius:6px;margin:16px 0}}
 .banner.warning{{background:#ffe7d9;border:1px solid #f5a623;color:#7a2d00}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{text-align:left}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Skill Report</h1>
  <div class="overall"><b>Overall:</b> {overall:.1f}</div>
  {incomplete_html}

  <table border='1' cellpadding='6' cellspacing='0'>
    <thead><tr><th>Domain</th><th>Score</th><th>SE</th><th>N</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {foot}

  <h3>Component breakdown (0–100)</h3>
  <table border='1' cellpadding='6' cellspacing='0'>
    <thead><tr><th>Domain</th><th>OBJ%</th><th>OPEN%</th></tr></thead>
    <tbody>{rows2}</tbody>
  </table>

  {plan_section}

  {open_summary}

  {reliability_html}

  {''.join(uv) if uv else ''}

  {cats}

  <p><b>Timing:</b> SR {summ.get('avg_rt_sr',0)}s · MCQ {summ.get('avg_rt_mcq',0)}s · SJT {summ.get('avg_rt_sjt',0)}s · OPEN N={summ.get('oe_items',0)}</p>
  {audit_links}
  {'<p><b>Reliability:</b> Some self-report answers need review.</p>' if reliability_flagged else ''}
</div>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
