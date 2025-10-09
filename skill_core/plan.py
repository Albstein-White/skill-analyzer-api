from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from . import config as cfg_defaults

log = logging.getLogger(__name__)


_TIER_ORDER: Dict[str, int] = {
    "F": 0,
    "D": 1,
    "C": 2,
    "B": 3,
    "A": 4,
    "S": 5,
    "SS": 6,
}

_DEFAULT_CONCEPTS: Dict[str, Sequence[str]] = {
    "Analytical": ("data comparisons", "ratio reasoning"),
    "Mathematical": ("algebra refresh", "multi-step calculations"),
    "Verbal": ("tone shifts", "evidence mapping"),
    "Memory": ("mnemonic chains", "chunking patterns"),
    "Spatial": ("rotation drills", "perspective shifts"),
    "Creativity": ("idea expansion", "divergent prompts"),
    "Strategy": ("scenario planning", "trade-off mapping"),
    "Social": ("empathy cues", "influence levers"),
}

_DEFAULT_RESOURCES: Dict[str, str] = {
    "Analytical": "Use weekly case sets and chart summaries to reinforce comparisons.",
    "Mathematical": "Schedule two short sessions with the SkillTier problem generator to review formulas.",
    "Verbal": "Summarise two editorials aloud, focusing on evidence statements.",
    "Memory": "Build a spaced-repetition deck for current projects.",
    "Spatial": "Sketch three perspective drawings to solidify rotation intuition.",
    "Creativity": "Run a 10-minute idea sprint with SCAMPER prompts twice this week.",
    "Strategy": "Deconstruct a recent decision memo and note three alternative plays.",
    "Social": "Role-play two stakeholder conversations focusing on active listening cues.",
}


@dataclass(frozen=True)
class PlanSettings:
    enabled: bool
    long_only: bool
    focus_max: int
    bullets_min: int
    bullets_max: int
    min_items_long: int
    se_target_long: float
    llm_enabled: bool
    latency_buckets: Tuple[float, float]

    @staticmethod
    def from_cfg(cfg: Mapping[str, Any] | None) -> "PlanSettings":
        def _cfg_value(name: str, default: Any) -> Any:
            if cfg is None:
                return default
            if isinstance(cfg, Mapping) and name in cfg:
                return cfg[name]
            return getattr(cfg, name, default) if hasattr(cfg, name) else default

        buckets = _cfg_value("PLAN_LATENCY_BUCKETS", cfg_defaults.PLAN_LATENCY_BUCKETS)
        if isinstance(buckets, Sequence) and len(buckets) >= 2:
            latency_buckets = (float(buckets[0]), float(buckets[1]))
        else:
            latency_buckets = cfg_defaults.PLAN_LATENCY_BUCKETS

        return PlanSettings(
            enabled=bool(_cfg_value("PLAN_ENABLED", cfg_defaults.PLAN_ENABLED)),
            long_only=bool(_cfg_value("PLAN_LONG_ONLY", cfg_defaults.PLAN_LONG_ONLY)),
            focus_max=int(_cfg_value("PLAN_FOCUS_MAX", cfg_defaults.PLAN_FOCUS_MAX)),
            bullets_min=int(_cfg_value("PLAN_BULLETS_MIN", cfg_defaults.PLAN_BULLETS_MIN)),
            bullets_max=int(_cfg_value("PLAN_BULLETS_MAX", cfg_defaults.PLAN_BULLETS_MAX)),
            min_items_long=int(_cfg_value("PLAN_MIN_ITEMS_LONG", cfg_defaults.PLAN_MIN_ITEMS_LONG)),
            se_target_long=float(_cfg_value("PLAN_SE_TARGET_LONG", cfg_defaults.PLAN_SE_TARGET_LONG)),
            llm_enabled=bool(_cfg_value("PLAN_LLM_ENABLED", cfg_defaults.PLAN_LLM_ENABLED)),
            latency_buckets=latency_buckets,
        )


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}
def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _tier_rank(value: Any) -> int:
    tier = str(value or "").upper().strip()
    return _TIER_ORDER.get(tier, len(_TIER_ORDER))


def _clamp_level(level: int) -> int:
    lo = getattr(cfg_defaults, "LEVEL_MIN", -2)
    hi = getattr(cfg_defaults, "LEVEL_MAX", 2)
    return max(lo, min(hi, int(level)))


def _extract_concept_tag(domain: str, tags: Iterable[Any]) -> Tuple[str, bool]:
    for raw in tags or []:
        if isinstance(raw, str) and raw.strip():
            return raw.strip(), True
        if isinstance(raw, Mapping):
            for key in ("tag", "label", "name"):
                val = raw.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip(), True
    fallback = _DEFAULT_CONCEPTS.get(domain)
    if fallback:
        return fallback[0], False
    return "core patterns", False


def _timing_targets(
    domain_stats: Mapping[str, Any],
    summary: Mapping[str, Any],
    settings: PlanSettings,
) -> Tuple[int, int, int]:
    stats = domain_stats.get("latency_stats") if isinstance(domain_stats, Mapping) else None
    low = high = quick = 0
    if isinstance(stats, Mapping) and stats:
        low = round(max(1.0, _safe_float(stats.get("p25"), 0.0)))
        high = round(max(low + 1.0, _safe_float(stats.get("p75"), 0.0)))
        quick = round(max(1.0, _safe_float(stats.get("p10"), low * 0.6)))
    else:
        avg_rt = _safe_float(summary.get("avg_rt_mcq"), 25.0)
        low = round(max(1.0, avg_rt * settings.latency_buckets[0]))
        high = round(max(low + 1.0, avg_rt * settings.latency_buckets[1]))
        quick = round(max(1.0, low * 0.6))
    if quick >= low:
        quick = max(1, low - 2)
    return max(1, low), max(low + 1, high), max(1, quick)


def _build_domain_plan(
    domain_stats: Mapping[str, Any],
    summary: Mapping[str, Any],
    settings: PlanSettings,
) -> Dict[str, Any]:
    domain = str(domain_stats.get("domain") or "Domain")
    theta = _safe_float(domain_stats.get("theta"), 0.0)
    b_stable = _safe_int(domain_stats.get("b_stable"), 0)
    level_val = _clamp_level(max(b_stable, round(theta)))
    level_label = f"{level_val:+d}"

    accuracy_map = domain_stats.get("accuracy_by_level") if isinstance(domain_stats, Mapping) else {}
    level_acc = 0.0
    if isinstance(accuracy_map, Mapping):
        if level_val in accuracy_map:
            level_acc = _safe_float(accuracy_map.get(level_val), 0.0)
        elif str(level_val) in accuracy_map:
            level_acc = _safe_float(accuracy_map.get(str(level_val)), 0.0)

    items_map = domain_stats.get("items_by_level") if isinstance(domain_stats, Mapping) else {}
    level_seen = 0
    if isinstance(items_map, Mapping):
        if level_val in items_map:
            level_seen = _safe_int(items_map.get(level_val), 0)
        elif str(level_val) in items_map:
            level_seen = _safe_int(items_map.get(str(level_val)), 0)

    drill_text = (
        f"Practice level {level_label} sets with deliberate review. "
        f"Baseline: {level_acc*100:.0f}% over {level_seen} items."
        if level_seen
        else f"Practice level {level_label} sets with deliberate review."
    )
    drill_goal = {
        "type": "drill",
        "text": drill_text,
        "measure": "Goal: ≥8/10 correct twice this week.",
    }

    concept_tag, concept_specific = _extract_concept_tag(
        domain, domain_stats.get("missed_tags") if isinstance(domain_stats, Mapping) else []
    )
    concept_goal = {
        "type": "concept",
        "tag": concept_tag,
        "text": (
            f"Design a micro-review on {concept_tag} and teach it back in 10 minutes."
            if concept_specific
            else f"Run a focused refresher on {concept_tag}; capture three takeaways."
        ),
    }

    low_t, high_t, quick_t = _timing_targets(domain_stats, summary, settings)
    timing_goal = {
        "type": "timing",
        "text": (
            f"Answer level {level_label} items between {low_t}-{high_t}s; avoid guesses faster than {quick_t}s."
        ),
    }

    next_level = _clamp_level(level_val + 1)
    if next_level == level_val:
        challenge_text = (
            f"Sustain level {level_label} mastery with weekly mixed sets; pass = 3/4 correct."
        )
    else:
        challenge_label = f"{next_level:+d}"
        challenge_text = (
            f"Attempt a level {challenge_label} mini-set after each drill; pass = 3/4 correct."
        )
    challenge_goal = {
        "type": "challenge",
        "text": challenge_text,
        "measure": "Document outcomes in a shared log.",
    }

    resource_text = _DEFAULT_RESOURCES.get(
        domain, "Use the SkillTier practice generator for two targeted sessions."
    )
    resource_goal = {
        "type": "resource",
        "text": resource_text,
    }

    goals: List[Dict[str, Any]] = [
        drill_goal,
        concept_goal,
        timing_goal,
        challenge_goal,
        resource_goal,
    ]

    if settings.bullets_max < len(goals):
        goals = goals[: settings.bullets_max]
    if len(goals) < settings.bullets_min:
        goals.extend(goals[: max(0, settings.bullets_min - len(goals))])

    goals = _maybe_rewrite_with_llm(domain, level_label, goals, settings)

    if len(goals) > settings.bullets_max:
        goals = goals[: settings.bullets_max]

    if len(goals) < settings.bullets_min:
        seed = goals or [drill_goal]
        idx = 0
        while len(goals) < settings.bullets_min:
            goals.append(dict(seed[idx % len(seed)]))
            idx += 1

    return {
        "domain": domain,
        "level": level_label,
        "goals": goals,
    }


def _maybe_rewrite_with_llm(
    domain: str,
    level_label: str,
    goals: List[Dict[str, Any]],
    settings: PlanSettings,
) -> List[Dict[str, Any]]:
    if not settings.llm_enabled:
        return goals
    try:
        from . import llm_bridge

        backend = llm_bridge.backend_in_use()
        if backend != "azure":
            return goals

        from .azure_cfg import client as azure_client

        payload = {"domain": domain, "level": level_label, "goals": goals}
        prompt = (
            "You are a coaching assistant. Rewrite the goals to sound supportive, clear, and time-bound. "
            "Preserve each goal's type/tag/measure fields. Return ONLY JSON matching the input structure.\n"
            f"Input: {json.dumps(payload, ensure_ascii=False)}"
        )
        cli = azure_client()
        resp = cli.chat.completions.create(
            model=llm_bridge.azure_settings().deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite improvement plans. Respond strictly with valid JSON matching the input schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            top_p=0.9,
            max_tokens=500,
        )
        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            return goals
        data = json.loads(content)
        new_goals = data.get("goals") if isinstance(data, dict) else data
        parsed: List[Dict[str, Any]] = []
        for g in new_goals or []:
            if isinstance(g, Mapping) and "type" in g and "text" in g:
                parsed.append(dict(g))
        return parsed or goals
    except Exception as exc:  # pragma: no cover - optional path
        log.debug("plan LLM fallback: %s", exc)
        return goals


def _domain_sort_key(domain_stats: Mapping[str, Any]) -> Tuple[int, float, int]:
    tier_rank = _tier_rank(domain_stats.get("tier"))
    se_val = _safe_float(domain_stats.get("se", domain_stats.get("rasch_se")), 1.0)
    b_stable = _safe_int(domain_stats.get("b_stable"), 0)
    return (tier_rank, -se_val, b_stable)


def _domain_se(domain_stats: Mapping[str, Any]) -> float:
    return _safe_float(domain_stats.get("se", domain_stats.get("rasch_se")), 1.0)


def _total_items(domains: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> int:
    total = 0
    for dom in domains:
        if not isinstance(dom, Mapping):
            continue
        if "n" in dom:
            total += _safe_int(dom.get("n"), 0)
        else:
            obj = _safe_int(dom.get("asked_obj"), 0)
            open_n = _safe_int(dom.get("asked_open"), 0)
            total += obj + open_n
    if total <= 0 and isinstance(summary, Mapping):
        total = _safe_int(summary.get("total_items"), 0)
    return max(0, total)


def generate_plan(result: Any, cfg: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    """Generate a SMART improvement plan for long adaptive runs.

    Parameters
    ----------
    result:
        Finalized result payload (dict or Result dataclass) including `domain_scores` and `summary`.
    cfg:
        Configuration mapping (typically output of :func:`skill_core.config.load_config`).

    Returns
    -------
    list of dict
        List of domain plans (each containing domain, level, goals). Empty when disabled.
    """

    settings = PlanSettings.from_cfg(cfg)
    if not settings.enabled:
        return []

    result_dict = _as_dict(result)
    run_type = str(
        result_dict.get("run_type")
        or (result_dict.get("meta") or {}).get("run")
        or (result_dict.get("meta") or {}).get("kind")
        or ""
    ).lower()

    if settings.long_only and run_type != "long":
        return []

    domains_raw = result_dict.get("domain_scores") or []
    domains: List[Dict[str, Any]] = [_as_dict(d) for d in domains_raw if d is not None]
    if not domains:
        return []

    summary = result_dict.get("summary") or {}
    total_items = _total_items(domains, summary)

    focus_max = max(1, settings.focus_max)
    if run_type == "long" and total_items and total_items < settings.min_items_long:
        focus_max = min(focus_max, 2)

    domains_sorted = sorted(domains, key=_domain_sort_key)

    priority = [d for d in domains_sorted if _domain_se(d) >= settings.se_target_long]
    fallback = [d for d in domains_sorted if d not in priority]
    ordered = priority + fallback

    focus_count = min(len(ordered), focus_max)
    if focus_count < 2 and len(ordered) >= 2:
        focus_count = 2
    ordered = ordered[:focus_count]

    plan = [
        _build_domain_plan(domain_stats, summary, settings)
        for domain_stats in ordered
    ]

    return plan


def _demo_plan() -> None:  # pragma: no cover - developer aid
    sample_result = {
        "run_type": "long",
        "summary": {
            "avg_rt_mcq": 28.0,
            "total_items": 96,
        },
        "domain_scores": [
            {
                "domain": "Analytical",
                "tier": "C",
                "theta": -0.2,
                "se": 0.32,
                "b_stable": 0,
                "accuracy_by_level": {0: 0.55},
                "items_by_level": {0: 9},
                "latency_stats": {"p10": 8.0, "p25": 15.0, "p50": 24.0, "p75": 36.0},
            },
            {
                "domain": "Verbal",
                "tier": "B",
                "theta": 0.15,
                "se": 0.41,
                "b_stable": 0,
                "accuracy_by_level": {0: 0.62},
                "items_by_level": {0: 11},
            },
            {
                "domain": "Strategy",
                "tier": "A",
                "theta": 0.8,
                "se": 0.27,
                "b_stable": 1,
                "accuracy_by_level": {1: 0.67},
                "items_by_level": {1: 12},
            },
        ],
    }

    plan = generate_plan(sample_result, None)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":  # pragma: no cover - developer diagnostics
    _demo_plan()

