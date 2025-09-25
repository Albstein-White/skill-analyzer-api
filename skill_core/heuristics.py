# skill_core/heuristics.py
from __future__ import annotations
import re

_METRIC_RX     = re.compile(r'\b(\d+%|percent|metric|kpi|conversion|retention|nps|mau|roi|rmse|mse|auc|precision|recall|accuracy|latency|time|error|throughput)\b', re.I)
_TIMELINE_RX   = re.compile(r'\b(min|mins|minute|minutes|hour|hours|daily|weekly|day|week|month|timeline|schedule|timebox|time-box|D\d|deadline|by\s+\d{4}-\d{2}-\d{2})\b', re.I)
_VALIDATION_RX = re.compile(r'\b(holdout|validation|cross[- ]?val|cross[- ]?validation|ab[- ]?test|a/b|control|baseline|residual|z-?score|iqr|p[- ]?value|confidence|significance)\b', re.I)
_DECISION_RX   = re.compile(r'\b(decision rule|threshold|gate|go/?no[- ]?go|success criterion|stopping rule|cutoff|only if)\b', re.I)

def heuristic_open_score(text: str) -> float:
    if not isinstance(text, str): return 0.0
    t = text.strip()
    if not t: return 0.0

    score = 0.40
    has_metric     = bool(_METRIC_RX.search(t))
    has_timeline   = bool(_TIMELINE_RX.search(t))
    has_validation = bool(_VALIDATION_RX.search(t))
    has_decision   = bool(_DECISION_RX.search(t))

    if has_metric:     score += 0.20
    if has_timeline:   score += 0.15
    if has_validation: score += 0.15
    if has_decision:   score += 0.10

    wc = max(1, len(t.split()))
    if 12 <= wc <= 80: score += 0.05
    if (has_metric + has_timeline + has_validation + has_decision) >= 3: score += 0.12

    return max(0.0, min(1.0, score))
