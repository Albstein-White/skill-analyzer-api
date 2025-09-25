# skill_core/reporting.py
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any, Dict

from .calibration import apply_to_result

# -------- utils: make any object JSON-safe ----------
def _to_basic(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _to_basic(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_basic(v) for v in x]
    # common escape hatches
    if hasattr(x, "to_dict"):
        try:
            return _to_basic(x.to_dict())  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(x, "__dict__"):
        try:
            return _to_basic(vars(x))
        except Exception:
            pass
    return str(x)

# -------- minimal HTML rendering (independent of report_html) ----------
def _render_html(data: Dict[str, Any], title: str = "Skill Report") -> str:
    ds = data.get("domain_summary") or data.get("domains") or {}
    rows = []
    for dom, val in ds.items():
        if isinstance(val, dict):
            score = val.get("score", val.get("S", val.get("value", "")))
            se    = val.get("se", "")
            n     = val.get("n", "")
        else:
            score, se, n = val, "", ""
        try:
            score_txt = f"{float(score):.2f}"
        except Exception:
            score_txt = str(score)
        se_txt = f"{float(se):.2f}" if se not in ("", None) and isinstance(se, (int, float)) else ""
        n_txt  = f"{int(n)}" if isinstance(n, (int, float)) and not math.isnan(float(n)) else ""
        rows.append(f"<tr><td>{dom}</td><td>{score_txt}</td><td>{se_txt}</td><td>{n_txt}</td></tr>")

    overall = data.get("overall")
    if isinstance(overall, dict):
        overall_txt = f"{overall.get('score', '')}"
    else:
        overall_txt = str(overall) if overall is not None else ""

    table = (
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<thead><tr><th>Domain</th><th>Score</th><th>SE</th><th>N</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    meta = data.get("calibration", {})
    calib = f"<p><b>Calibration:</b> {json.dumps(_to_basic(meta))}</p>" if meta else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
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
  <h1>{title}</h1>
  <div class="overall"><b>Overall:</b> {overall_txt}</div>
  {table}
  {calib}
</div>
</body>
</html>"""

# -------- public API expected by autoplay.py ----------
def write_report(result: Any, out_path: str, title: str = "Skill Report") -> str:
    """
    Keeps the signature your autoplay.py expects.
    - Converts to plain dict
    - Applies calibration (anchors/slack)
    - Writes HTML to out_path
    - Also writes JSON next to it for debugging
    Returns out_path.
    """
    # 1) convert & calibrate
    basic = _to_basic(result)
    calibrated = apply_to_result(basic)  # safe if already dict-like

    # 2) ensure folders
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 3) HTML
    html = _render_html(calibrated, title=title)
    out.write_text(html, encoding="utf-8")

    # 4) JSON sidecar
    sidecar = out.with_suffix(".json")
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(_to_basic(calibrated), f, ensure_ascii=False, indent=2)

    return str(out)
