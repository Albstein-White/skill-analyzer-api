# skill_core/engine.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone
import random, os, json, csv, time, logging, re, math
from pathlib import Path

from .types import Item, Answer, DomainScore, HiddenSkill, Result
from .question_bank import load_bank, DOMAINS
from .scoring import score_item
from .validators import count_traps, consistency_index
from .rarity import tier, rarity_label
from .config import (
    load_config,
    seed_rng,
    RASCH_PRIOR_VAR,
    RASCH_ETA,
    PASS_RULE_SHORT,
    PASS_RULE_LONG,
    DEMOTE_RULE,
    SE_TARGET_SHORT,
    SE_TARGET_LONG,
    OBJ_MIN_SHORT,
    OBJ_MAX_SHORT,
    OBJ_MIN_LONG,
    OBJ_MAX_LONG,
    CAP_SHORT,
    CAP_LONG,
    LEVEL_MIN,
    LEVEL_MAX,
    A_OPEN,
    ETA_OPEN,
    DELTA_OPEN_MAX,
    OPEN_INFO_CAP_RATIO,
    OPEN_MIN_WORDS,
    OPEN_MIN_RUBRIC,
    OPEN_LATENCY_P10_MS,
    SR_START_SHIFT,
    DEBUG_TRACE,
    TRACE_FIELDS,
)
from .policy import QuestionPolicy, PolicyState, DomainHistory
from . import rasch


log = logging.getLogger(__name__)


def _emit_trace(**values: object) -> None:
    if not DEBUG_TRACE:
        return
    ordered = []
    for key in TRACE_FIELDS:
        if key in values:
            ordered.append(f"{key}={values[key]}")
    if ordered:
        log.info("trace %s", " ".join(str(val) for val in ordered))


def _default_level_stats() -> Dict[int, Dict[str, int]]:
    return {lvl: {"seen": 0, "correct": 0} for lvl in range(-2, 3)}


_RECENT_WINDOW_LIMIT = max(
    PASS_RULE_SHORT["window"], PASS_RULE_LONG["window"], DEMOTE_RULE["window"]
)


def _clamp_level(level: int) -> int:
    return max(LEVEL_MIN, min(LEVEL_MAX, int(level)))


_CALIBRATION_V2_CACHE: Optional[Dict[str, object]] = None


def _load_calibration_v2() -> Dict[str, object]:
    """Lazy-load calibration metadata used to map θ to normed scores."""

    global _CALIBRATION_V2_CACHE
    if _CALIBRATION_V2_CACHE is None:
        path = Path(__file__).with_name("calibration_v2.json")
        try:
            with path.open("r", encoding="utf-8") as f:
                _CALIBRATION_V2_CACHE = json.load(f)
        except Exception:
            _CALIBRATION_V2_CACHE = {
                "floor": 10.0,
                "ceil": 100.0,
                "min_span": 40.0,
                "domains": {},
            }
    return _CALIBRATION_V2_CACHE


def _theta_to_norm_score(theta: float, domain: str) -> float:
    """Convert Rasch θ into the calibrated 0–100 domain score."""

    calib = _load_calibration_v2()
    base_prob = rasch.sigma(theta)
    base_scale = base_prob * 10.0  # 0..10 ability anchor

    floor = float(calib.get("floor", 10.0))
    ceil = float(calib.get("ceil", 100.0))
    min_span = float(calib.get("min_span", 40.0))
    domain_cfg = calib.get("domains", {}).get(domain, {})
    worst = float(domain_cfg.get("worst", floor))
    best = float(domain_cfg.get("best", ceil))

    if best <= worst:
        best = worst + min_span

    span = max(best - worst, min_span)
    span = min(span, max(ceil - floor, min_span))

    score = worst + (base_scale / 10.0) * span
    return float(max(floor, min(ceil, score)))


def _sr_shift_from_mean(mean: Optional[float]) -> int:
    if mean is None:
        return 0
    if mean <= 2.5:
        return int(SR_START_SHIFT.get("low", -1))
    if mean >= 4.5:
        return int(SR_START_SHIFT.get("high", 1))
    return int(SR_START_SHIFT.get("mid", 0))


def _is_negative_sr_item(item: Item) -> bool:
    pol = getattr(item, "polarity", None)
    if isinstance(pol, str) and pol.lower().startswith("neg"):
        return True
    iid = getattr(item, "id", None)
    return isinstance(iid, str) and iid.endswith("_neg")


def _sr_normalized_value(item: Item, answer: Answer) -> Optional[int]:
    try:
        raw = int(getattr(answer, "value", 0))
    except Exception:
        return None
    if raw < 0:
        raw = 0
    if raw > 4:
        raw = 4
    likert = raw + 1
    if _is_negative_sr_item(item):
        likert = 6 - likert
    return likert


def _sr_trap_failed(item: Item, answer: Answer) -> bool:
    if not getattr(item, "is_trap", False):
        return False
    try:
        val = int(getattr(answer, "value", 0))
    except Exception:
        return False
    flag = getattr(item, "trap_flag_index", None)
    if flag is not None:
        try:
            return int(val) == int(flag)
        except Exception:
            return False
    return int(val) == 5


def _run_parameters(run_type: str) -> tuple[dict, float, int, int]:
    if run_type == "short":
        return PASS_RULE_SHORT, SE_TARGET_SHORT, OBJ_MIN_SHORT, OBJ_MAX_SHORT
    return PASS_RULE_LONG, SE_TARGET_LONG, OBJ_MIN_LONG, OBJ_MAX_LONG


def _apply_objective_step(
    domain_state: DomainState,
    level_bucket: int,
    correct: bool,
    run_type: str,
    target_level: Optional[int] = None,
) -> Dict[str, object]:
    """Apply Rasch update and ladder progression for an objective answer."""

    prev_level = int(domain_state.level)
    bucket = _clamp_level(level_bucket)

    theta_before = domain_state.theta
    theta_after = rasch.map_update(
        theta_before,
        bucket,
        bool(correct),
        a=1.0,
        prior_var=RASCH_PRIOR_VAR,
        eta=RASCH_ETA,
    )
    info_delta = rasch.item_info(theta_after, bucket, a=1.0)

    domain_state.theta = theta_after
    domain_state.info_total = max(domain_state.info_total + info_delta, 1e-6)
    domain_state.obj_info_total = max(domain_state.obj_info_total + info_delta, 0.0)
    domain_state.se = rasch.se_from_info(domain_state.info_total + domain_state.open_info_total)
    domain_state.obj_count += 1

    stats = domain_state.level_stats.setdefault(bucket, {"seen": 0, "correct": 0})
    stats["seen"] += 1
    stats["correct"] += int(correct)

    window_level = prev_level if target_level is None else int(target_level)
    same_level = window_level == prev_level
    if same_level:
        domain_state.recent_window.append(bool(correct))
        if len(domain_state.recent_window) > _RECENT_WINDOW_LIMIT:
            domain_state.recent_window.pop(0)
        domain_state.level_entry_seen += 1
    else:
        domain_state.recent_window = [bool(correct)]
        domain_state.level_entry_seen = 1

    pass_rule, _, _, _ = _run_parameters(run_type)
    new_level = prev_level
    promoted = False
    if same_level:
        window = domain_state.recent_window[-pass_rule["window"] :]
        if len(window) >= pass_rule["window"] and sum(window) >= pass_rule["need"]:
            new_level = _clamp_level(prev_level + 1)
            domain_state.b_stable = max(domain_state.b_stable, prev_level)
            domain_state.recent_window = []
            domain_state.level_entry_seen = 0
            promoted = True

    if same_level and not promoted:
        demo_window = domain_state.recent_window[-DEMOTE_RULE["window"] :]
        if len(demo_window) >= DEMOTE_RULE["window"]:
            misses = DEMOTE_RULE["window"] - sum(demo_window)
            if misses >= DEMOTE_RULE["need"] and domain_state.level_entry_seen > 1:
                new_level = _clamp_level(prev_level - 1)
                domain_state.recent_window = []
                domain_state.level_entry_seen = 0

    domain_state.level = new_level
    if domain_state.obj_count == 1:
        domain_state.b_peak = new_level
    else:
        domain_state.b_peak = max(domain_state.b_peak, new_level)

    return {
        "theta_before": theta_before,
        "theta_after": theta_after,
        "info_delta": info_delta,
        "level_before": prev_level,
        "level_after": new_level,
        "window": list(domain_state.recent_window),
        "stats": dict(stats),
        "se": domain_state.se,
    }


def _apply_open_step(
    domain_state: DomainState,
    difficulty: int,
    score: float,
    word_count: int,
    latency_ms: float,
    rubric_conf: Optional[float],
) -> Dict[str, object]:
    """Apply bounded Rasch-style residual update for an OPEN response."""

    bucket = max(min(int(difficulty), 1), -1)
    theta_before = domain_state.theta
    mu = rasch.sigma(theta_before - bucket)

    eta = ETA_OPEN
    eta_halved = False
    ignored_reason: Optional[str] = None
    scaled = False
    dtheta = 0.0
    open_info_delta = 0.0

    if latency_ms < float(OPEN_LATENCY_P10_MS):
        ignored_reason = "latency"
    else:
        if word_count < OPEN_MIN_WORDS or (
            rubric_conf is not None and rubric_conf < OPEN_MIN_RUBRIC
        ):
            eta *= 0.5
            eta_halved = True

        info_mu = mu * (1.0 - mu)
        denom = max(info_mu, 1e-6)
        dtheta = eta * (score - mu) / denom
        dtheta = max(min(dtheta, DELTA_OPEN_MAX), -DELTA_OPEN_MAX)

        open_info_delta = (A_OPEN * A_OPEN) * info_mu
        open_total_candidate = domain_state.open_info_total + open_info_delta
        cap = OPEN_INFO_CAP_RATIO * max(domain_state.obj_info_total, 1e-6)

        if open_total_candidate > cap and cap > 0.0:
            scale = cap / open_total_candidate if open_total_candidate > 0.0 else 0.0
            scale = max(min(scale, 1.0), 0.0)
            if scale < 1.0:
                dtheta *= scale
                open_info_delta *= scale
                scaled = True

        domain_state.theta = theta_before + dtheta
        domain_state.open_info_total += open_info_delta
        domain_state.open_info_total = min(domain_state.open_info_total, cap)
        domain_state.se = rasch.se_from_info(
            domain_state.info_total + domain_state.open_info_total
        )

    domain_state.open_count += 1
    entry: Dict[str, object] = {
        "b": bucket,
        "r": float(score),
        "mu": float(mu),
        "dtheta": float(dtheta if ignored_reason is None else 0.0),
        "scaled": bool(scaled),
    }
    if ignored_reason is not None:
        entry["ignored"] = ignored_reason
    if eta_halved:
        entry["eta_halved"] = True
    domain_state.open_contrib.append(entry)

    return {
        "theta_before": theta_before,
        "theta_after": domain_state.theta,
        "b": bucket,
        "mu": mu,
        "dtheta": entry["dtheta"],
        "scaled": scaled,
        "ignored": ignored_reason,
        "word_count": word_count,
        "latency_ms": latency_ms,
        "eta_halved": eta_halved,
        "open_info_delta": open_info_delta,
    }

# Default response-time baselines (seconds)
DEFAULT_BASE_RT = {"MCQ": 20.0, "SJT": 25.0, "SR": 10.0, "OPEN": 90.0}

@dataclass
class DomainState:
    mcq_correct: int = 0
    mcq_total:   int = 0
    sjt_sum:     float = 0.0
    sjt_total:   int = 0
    open_sum:    float = 0.0
    open_total:  int = 0
    sr_sum:      float = 0.0
    sr_total:    int = 0
    max_diff: int = -3
    theta: float = 0.0
    info_total: float = 1.0
    se: float = 1.0
    level: int = 0
    level_stats: Dict[int, Dict[str, int]] = field(default_factory=_default_level_stats)
    recent_window: List[bool] = field(default_factory=list)
    b_peak: int = 0
    b_stable: int = 0
    obj_count: int = 0
    sr_count: int = 0
    open_count: int = 0
    open_contrib: List[Dict[str, object]] = field(default_factory=list)
    obj_done: bool = False
    level_entry_seen: int = 0
    obj_info_total: float = 0.0
    open_info_total: float = 0.0
    sr_trap_count: int = 0
    sr_mirror_ok: bool = True

    def to_dict(self) -> Dict[str, object]:
        """JSON-friendly representation used for persistence/debugging."""

        return {
            "mcq_correct": self.mcq_correct,
            "mcq_total": self.mcq_total,
            "sjt_sum": self.sjt_sum,
            "sjt_total": self.sjt_total,
            "open_sum": self.open_sum,
            "open_total": self.open_total,
            "sr_sum": self.sr_sum,
            "sr_total": self.sr_total,
            "max_diff": self.max_diff,
            "theta": self.theta,
            "info_total": self.info_total,
            "se": self.se,
            "level": self.level,
            "level_stats": self.level_stats,
            "recent_window": list(self.recent_window),
            "b_peak": self.b_peak,
            "b_stable": self.b_stable,
            "obj_count": self.obj_count,
            "sr_count": self.sr_count,
            "open_count": self.open_count,
            "open_contrib": list(self.open_contrib),
            "obj_done": self.obj_done,
            "level_entry_seen": self.level_entry_seen,
            "obj_info_total": self.obj_info_total,
            "open_info_total": self.open_info_total,
            "sr_trap_count": self.sr_trap_count,
            "sr_mirror_ok": self.sr_mirror_ok,
        }

@dataclass
class EngineState:
    domains: Dict[str, DomainState] = field(default_factory=lambda: {d: DomainState() for d in DOMAINS})
    answers: Dict[str, Answer] = field(default_factory=dict)
    rt_sums: Dict[str, float] = field(default_factory=lambda: {"SR": 0.0, "MCQ": 0.0, "SJT": 0.0, "OPEN": 0.0})
    rt_counts: Dict[str, int] = field(default_factory=lambda: {"SR": 0, "MCQ": 0, "SJT": 0, "OPEN": 0})
    item_rows: List[Dict] = field(default_factory=list)  # per-item audit rows
    audit_events: List[Dict[str, object]] = field(default_factory=list)

def _safe_frac(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0

def _obj_frac(st: DomainState) -> float:
    mcq = _safe_frac(st.mcq_correct, st.mcq_total)
    sjt = _safe_frac(st.sjt_sum,     st.sjt_total)
    if st.mcq_total + st.sjt_total == 0:
        return 0.0
    return (mcq * st.mcq_total + sjt * st.sjt_total) / float(st.mcq_total + st.sjt_total)

def _open_frac(st: DomainState) -> float:
    return _safe_frac(st.open_sum, st.open_total)

def _sr_frac(st: DomainState) -> float:
    return _safe_frac(st.sr_sum, st.sr_total)

def _composite(domain: str, st: DomainState) -> tuple[float, float, dict]:
    obj = _obj_frac(st)
    opn = _open_frac(st)
    sr = _sr_frac(st)

    score = _theta_to_norm_score(st.theta, domain)
    parts = {"obj": round(100.0 * obj, 1), "open": round(100.0 * opn, 1), "sr": round(100.0 * sr, 1)}
    return score, st.se, parts

class AdaptiveSession:
    def __init__(self, run_type: str):
        assert run_type in ("short","long")
        self.cfg = load_config(); seed_rng(self.cfg)
        self.run_type = run_type
        self.state = EngineState()
        self.items: list[Item] = load_bank()
        random.shuffle(self.items)

        self.asked: set[str] = set()
        self.seen_variant_groups: set[str] = set()
        self._current: Optional[Item] = None
        self._step = 0
        self.last_domain_id: Optional[str] = None
        self._last_item_type: Optional[str] = None

        self._init_domain_levels()

        self.policy = QuestionPolicy(self.items, run_type)
        self._id_to_item: Dict[str, Item] = {it.id: it for it in self.items}
        self._mirror_map: Dict[str, str] = {}
        for it in self.items:
            if getattr(it, "type", "") == "SR":
                partner = getattr(it, "mirror_of", None)
                if isinstance(partner, str) and partner:
                    self._mirror_map[it.id] = partner
                    self._mirror_map.setdefault(partner, it.id)
        self._info_hist: List[float] = []

        # load RT baselines if present
        self.base_rt = DEFAULT_BASE_RT
        try:
            with open("rt_baseline.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            brt = {k: float(v) for k, v in data.items() if k in DEFAULT_BASE_RT and float(v) > 0}
            if len(brt) == len(DEFAULT_BASE_RT):
                self.base_rt = brt
        except Exception:
            pass

    def _init_domain_levels(self) -> None:
        """Initialize ladder starting levels based on SR means or priors."""

        sr_cfg = {}
        for key in ("sr_means", "sr_mean", "sr_baseline"):
            val = self.cfg.get(key)
            if isinstance(val, dict):
                sr_cfg = val
                break

        theta_cfg: Dict[str, float] = {}
        for key in ("theta_priors", "theta_prior", "domain_theta_prior"):
            val = self.cfg.get(key)
            if isinstance(val, dict):
                try:
                    theta_cfg = {k: float(v) for k, v in val.items()}
                except Exception:
                    theta_cfg = {}
                break

        for domain, st in self.state.domains.items():
            base_level = 0
            sr_mean: Optional[float] = None
            if st.sr_total > 0:
                sr_mean = st.sr_sum / max(1, st.sr_total)
            elif sr_cfg:
                raw = sr_cfg.get(domain)
                if raw is not None:
                    try:
                        sr_mean = float(raw)
                    except Exception:
                        sr_mean = None

            if sr_mean is not None:
                base_level += _sr_shift_from_mean(sr_mean)
            base_level = max(-1, min(1, base_level))

            start_level = base_level
            prior_val = theta_cfg.get(domain)
            if prior_val is not None:
                start_level = int(round(prior_val))

            start_level = _clamp_level(start_level)
            st.level = start_level
            st.recent_window.clear()
            st.level_entry_seen = 0
            st.b_peak = start_level
            if prior_val is not None:
                st.b_stable = max(st.b_stable, start_level)

    def _policy_state(self) -> PolicyState:
        hist: Dict[str, DomainHistory] = {}
        theta_map: Dict[str, float] = {}
        se_map: Dict[str, float] = {}
        for d in DOMAINS:
            dh = DomainHistory()
            dh.asked_ids = [iid for iid in self.asked if self._id_to_item[iid].domain == d]
            st = self.state.domains[d]
            theta_map[d] = st.theta
            se_map[d] = st.se
            dh.sr_count = st.sr_count
            dh.obj_count = st.obj_count
            dh.open_count = st.open_count
            denom = float(st.mcq_total + st.sjt_total)
            dh.obj_correct_frac = (st.mcq_correct + st.sjt_sum) / denom if denom > 0 else 0.0
            dh.level = st.level
            dh.level_stats = st.level_stats
            dh.recent_window = list(st.recent_window)
            dh.level_entry_seen = st.level_entry_seen
            dh.obj_done = st.obj_done
            dh.open_contrib = list(st.open_contrib)
            dh.obj_info_total = st.obj_info_total
            dh.open_info_total = st.open_info_total
            hist[d] = dh
        return PolicyState(
            run_type=self.run_type, theta=theta_map, se=se_map, asked=set(self.asked),
            seen_variant_groups=set(self.seen_variant_groups), step=self._step, hist=hist,
            info_history=self._info_hist, mirrored_domains_planned=getattr(self, "_mir_planned", set()),
            last_domain_id=self.last_domain_id,
            last_item_type=self._last_item_type,
        )

    def _update_obj_done(self, st: DomainState) -> None:
        _, se_target, obj_min, obj_max = _run_parameters(self.run_type)
        st.obj_done = (
            (st.obj_count >= obj_min and st.se <= se_target)
            or st.obj_count >= obj_max
        )

    def _update_sr_reliability(self, item: Item, answer: Answer, domain_state: DomainState) -> None:
        if _sr_trap_failed(item, answer):
            domain_state.sr_trap_count += 1

        partner_id = self._mirror_map.get(item.id)
        if not partner_id:
            return

        other_item = self._id_to_item.get(partner_id)
        other_answer = self.state.answers.get(partner_id)
        if other_item is None or other_answer is None:
            return

        current_val = _sr_normalized_value(item, answer)
        partner_val = _sr_normalized_value(other_item, other_answer)
        if current_val is None or partner_val is None:
            return

        if abs(current_val - partner_val) >= 3:
            domain_state.sr_mirror_ok = False

    def next_item(self) -> Optional[Item]:
        st = self._policy_state()
        if self.policy.should_stop(st): return None
        it = self.policy.next_item(st)
        self._current = it
        if it is not None:
            vg = getattr(it, "variant_group", None)
            if vg:
                self.seen_variant_groups.add(vg)
        return it

    def answer_current(self, answer: Answer) -> None:
        if not self._current: return
        it = self._current
        credit, meta = score_item(it, answer)

        ds = self.state.domains[it.domain]
        rt_ms = 0
        if answer.rt_sec is not None:
            rt_val = float(answer.rt_sec)
            self.state.rt_sums[it.type] += rt_val
            self.state.rt_counts[it.type] += 1
            try:
                rt_ms = max(0, int(round(rt_val * 1000.0)))
            except Exception:
                rt_ms = 0

        difficulty = int(getattr(it, "difficulty", 0) or 0)
        level_bucket = max(min(difficulty, 2), -2)
        is_obj = it.type in ("MCQ", "SJT")

        level_before_answer = int(ds.level)
        event_record: Optional[Dict[str, object]] = None
        timestamp = datetime.now(timezone.utc).isoformat()

        if is_obj:
            correct_bool = bool(credit >= 0.999) if it.type == "MCQ" else bool(credit >= 0.5)
            target_level = getattr(it, "_target_level", ds.level)
            metrics = _apply_objective_step(
                ds,
                level_bucket,
                correct_bool,
                self.run_type,
                target_level=target_level,
            )
            self._info_hist.append(ds.info_total + ds.open_info_total)

            self._update_obj_done(ds)

            log.debug(
                (
                    "rasch_update domain=%s item=%s b=%d correct=%s theta=%.4f->%.4f "
                    "info=%.4f se=%.4f level=%+d->%+d window=%s stats=%s"
                ),
                it.domain,
                it.id,
                level_bucket,
                int(correct_bool),
                metrics["theta_before"],
                metrics["theta_after"],
                metrics["info_delta"],
                metrics["se"],
                metrics["level_before"],
                metrics["level_after"],
                metrics["window"],
                metrics["stats"],
            )

            _emit_trace(
                domain=it.domain,
                item_id=it.id,
                type=it.type,
                level=ds.level,
                b=level_bucket,
                correct_or_r=int(correct_bool),
                theta_before=metrics["theta_before"],
                theta_after=metrics["theta_after"],
                se=ds.se,
                info_gain=metrics["info_delta"],
            )

            event_record = {
                "t": timestamp,
                "domain": it.domain,
                "item_id": it.id,
                "type": it.type,
                "b": int(level_bucket),
                "level_before": int(metrics["level_before"]),
                "level_after": int(metrics["level_after"]),
                "correct_or_r": 1.0 if correct_bool else 0.0,
                "theta_before": float(metrics["theta_before"]),
                "theta_after": float(metrics["theta_after"]),
                "se_after": float(ds.se),
                "latency_ms": int(rt_ms),
            }

        if it.type == "MCQ":
            ds.mcq_total += 1
            ds.mcq_correct += 1 if credit >= 0.999 else 0
            ds.max_diff = max(ds.max_diff, difficulty)
        elif it.type == "SJT":
            ds.sjt_total += 1
            ds.sjt_sum   += float(credit)
        elif it.type == "OPEN":
            theta_before_open = ds.theta
            ds.open_total += 1
            ds.open_sum   += float(credit)

            text_value = getattr(answer, "value", "")
            word_count = len(re.findall(r"\w+", str(text_value)))
            latency_ms = float(answer.rt_sec * 1000.0) if answer.rt_sec is not None else 0.0
            rubric_conf = None
            if isinstance(meta, dict):
                raw_conf = meta.get("rubric_conf")
                try:
                    rubric_conf = float(raw_conf) if raw_conf is not None else None
                except (TypeError, ValueError):
                    rubric_conf = None

            metrics = _apply_open_step(
                ds,
                level_bucket,
                float(credit),
                word_count,
                latency_ms,
                rubric_conf,
            )
            self._update_obj_done(ds)
            self._info_hist.append(ds.info_total + ds.open_info_total)

            log.debug(
                (
                    "open_update domain=%s item=%s b=%d r=%.2f mu=%.3f dtheta=%.4f "
                    "scaled=%s ignored=%s se=%.4f open_info=%.4f"
                ),
                it.domain,
                it.id,
                metrics["b"],
                float(credit),
                metrics["mu"],
                metrics["dtheta"],
                metrics["scaled"],
                metrics["ignored"],
                ds.se,
                ds.open_info_total,
            )

            _emit_trace(
                domain=it.domain,
                item_id=it.id,
                type=it.type,
                level=ds.level,
                b=metrics["b"],
                correct_or_r=float(credit),
                theta_before=theta_before_open,
                theta_after=ds.theta,
                se=ds.se,
                info_gain=metrics["open_info_delta"],
            )

            event_record = {
                "t": timestamp,
                "domain": it.domain,
                "item_id": it.id,
                "type": it.type,
                "b": int(metrics["b"]),
                "level_before": int(level_before_answer),
                "level_after": int(ds.level),
                "correct_or_r": float(credit),
                "theta_before": float(theta_before_open),
                "theta_after": float(ds.theta),
                "se_after": float(ds.se),
                "latency_ms": int(rt_ms),
            }
        elif it.type == "SR":
            theta_before_sr = ds.theta
            ds.sr_total += 1
            ds.sr_sum   += float(credit)
            ds.sr_count += 1
            self._update_sr_reliability(it, answer, ds)

            _emit_trace(
                domain=it.domain,
                item_id=it.id,
                type=it.type,
                level=ds.level,
                b=level_bucket,
                correct_or_r=float(credit),
                theta_before=theta_before_sr,
                theta_after=ds.theta,
                se=ds.se,
                info_gain=0.0,
            )

            event_record = {
                "t": timestamp,
                "domain": it.domain,
                "item_id": it.id,
                "type": it.type,
                "b": int(level_bucket),
                "level_before": int(level_before_answer),
                "level_after": int(ds.level),
                "correct_or_r": float(credit),
                "theta_before": float(theta_before_sr),
                "theta_after": float(ds.theta),
                "se_after": float(ds.se),
                "latency_ms": int(rt_ms),
            }

        # per-item audit row
        self.state.item_rows.append({
            "ts": round(time.time(), 3),
            "run_type": self.run_type,
            "domain": it.domain,
            "type": it.type,
            "id": it.id,
            "answer": getattr(answer, "value", None),
            "credit": float(credit),
            "rt_sec": float(answer.rt_sec if answer.rt_sec is not None else 0.0),
        })

        self.asked.add(it.id)
        vg = getattr(it, "variant_group", None)
        if vg:
            self.seen_variant_groups.add(vg)
        self.state.answers[it.id] = answer
        self._step += 1
        self.last_domain_id = it.domain
        self._last_item_type = it.type
        self._current = None

        if event_record is None:
            event_record = {
                "t": timestamp,
                "domain": it.domain,
                "item_id": it.id,
                "type": it.type,
                "b": int(level_bucket),
                "level_before": int(level_before_answer),
                "level_after": int(ds.level),
                "correct_or_r": float(credit),
                "theta_before": float(ds.theta),
                "theta_after": float(ds.theta),
                "se_after": float(ds.se),
                "latency_ms": int(rt_ms),
            }

        self.state.audit_events.append(event_record)

    def _summary_categories(self) -> Dict[str, float]:
        # Precision = objective accuracy overall
        obj_num = 0.0; obj_den = 0.0
        for d in DOMAINS:
            st = self.state.domains[d]
            n = (st.mcq_total + st.sjt_total)
            if n > 0:
                obj_num += (st.mcq_correct + st.sjt_sum)
                obj_den += n
        obj_acc = _safe_frac(obj_num, obj_den)
        precision = round(100.0 * obj_acc, 1)

        # Speed = normalized RT vs baselines, blended with accuracy
        comps = []
        for t in ("MCQ", "SJT", "SR", "OPEN"):
            c = self.state.rt_counts.get(t, 0)
            if c <= 0: continue
            avg_rt = self.state.rt_sums.get(t, 0.0) / max(1, c)
            base = float(self.base_rt.get(t, DEFAULT_BASE_RT[t]))
            ratio = base / max(1e-6, avg_rt)  # >1 = faster
            ratio = min(max(ratio, 0.0), 1.5)
            comps.append(ratio / 1.5)
        speed_component = sum(comps) / len(comps) if comps else 0.0
        speed = round(100.0 * (0.6 * speed_component + 0.4 * obj_acc), 1)

        # Consistency (0..100 from 0..1)
        cons = consistency_index([self._id_to_item[i] for i in self.asked if i in self._id_to_item], self.state.answers)
        try: cons_val = float(cons)
        except: cons_val = 0.0
        cons_val = min(max(cons_val, 0.0), 1.0)
        consistency = round(100.0 * cons_val, 1)

        return {"speed_score": speed, "precision_score": precision, "consistency_score": consistency}

    def _write_items_csv(self, out_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["ts","run_type","domain","type","id","answer","credit","rt_sec"])
                w.writeheader()
                for r in self.state.item_rows:
                    w.writerow(r)
        except Exception:
            pass

    def finalize(self) -> Result:
        asked = [self._id_to_item[iid] for iid in self.asked if iid in self._id_to_item]
        traps = count_traps(asked, self.state.answers)

        out_scores: List[DomainScore] = []
        for d in DOMAINS:
            st = self.state.domains[d]
            score, composite_se, parts = _composite(d, st)

            # A-cap when no OPEN answered
            cap_applied = None
            if st.open_total == 0 and score > 80.0:
                score = 80.0
                cap_applied = "A"

            reliability = {
                "sr_mirror_ok": bool(st.sr_mirror_ok),
                "trap_count": int(st.sr_trap_count),
            }
            ds = DomainScore(
                domain=d,
                theta=st.theta,
                se=st.se,
                norm_score=round(score, 1),
                tier=tier(score),
                rarity=rarity_label(score),
                reliability=reliability,
            )
            asked_obj = int(st.mcq_total + st.sjt_total); asked_open = int(st.open_total)
            setattr(ds, "asked_obj", asked_obj); setattr(ds, "asked_open", asked_open); setattr(ds, "n", asked_obj + asked_open)
            setattr(ds, "obj_pct", parts["obj"]); setattr(ds, "open_pct", parts["open"]); setattr(ds, "sr_pct", parts["sr"])
            if cap_applied: setattr(ds, "cap", cap_applied)
            setattr(ds, "info_total", st.info_total)
            setattr(ds, "level", st.level)
            setattr(ds, "level_stats", st.level_stats)
            setattr(ds, "b_peak", st.b_peak)
            setattr(ds, "b_stable", st.b_stable)
            setattr(ds, "obj_count", st.obj_count)
            setattr(ds, "open_count", st.open_count)
            setattr(ds, "rasch_se", st.se)
            setattr(ds, "composite_se", composite_se)
            setattr(ds, "open_contrib", list(st.open_contrib))
            setattr(ds, "obj_info_total", st.obj_info_total)
            setattr(ds, "open_info_total", st.open_info_total)

            items_by_level: Dict[int, int] = {}
            accuracy_by_level: Dict[int, float] = {}
            for lvl_key, stats in (st.level_stats or {}).items():
                try:
                    lvl_int = int(lvl_key)
                except Exception:
                    continue
                seen = int(stats.get("seen", 0)) if isinstance(stats, dict) else 0
                correct = int(stats.get("correct", 0)) if isinstance(stats, dict) else 0
                items_by_level[lvl_int] = max(seen, 0)
                accuracy_by_level[lvl_int] = _safe_frac(correct, seen)
            setattr(ds, "items_by_level", items_by_level)
            setattr(ds, "accuracy_by_level", accuracy_by_level)

            latencies = [
                float(row.get("rt_sec", 0.0) or 0.0)
                for row in self.state.item_rows
                if row.get("domain") == d and row.get("type") in {"MCQ", "SJT"}
            ]
            latencies = [max(0.0, lt) for lt in latencies if isinstance(lt, (int, float))]
            latency_stats: Dict[str, float] = {}
            if latencies:
                latencies.sort()

                def _quantile(q: float) -> float:
                    if not latencies:
                        return 0.0
                    idx = (len(latencies) - 1) * q
                    lower = int(math.floor(idx))
                    upper = int(math.ceil(idx))
                    if lower == upper:
                        return float(latencies[lower])
                    low_val = float(latencies[lower])
                    high_val = float(latencies[upper])
                    return low_val + (high_val - low_val) * (idx - lower)

                latency_stats = {
                    "avg": sum(latencies) / len(latencies),
                    "p10": _quantile(0.10),
                    "p25": _quantile(0.25),
                    "p50": _quantile(0.5),
                    "p75": _quantile(0.75),
                }
            setattr(ds, "latency_stats", latency_stats)
            out_scores.append(ds)

        top = [x.domain for x in sorted(out_scores, key=lambda x: x.norm_score, reverse=True)[:5]]

        cats = self._summary_categories()

        # Undervalued skills only (Obj - SR)
        hidden: List[HiddenSkill] = []
        for d, st in self.state.domains.items():
            obj = _obj_frac(st); sr = _sr_frac(st); gap = obj - sr
            if (st.mcq_total + st.sjt_total) >= 3 and st.sr_total >= 3 and gap >= 0.12:
                conf = "High" if gap>=0.20 else ("Medium" if gap>=0.16 else "Low")
                hidden.append(HiddenSkill(domain=d, confidence=conf, reason=f"Objective-SR gap {gap:.2f}"))
        hidden = hidden[:5]

        total_items = 0
        for d in DOMAINS:
            st = self.state.domains[d]
            total_items += int(st.mcq_total + st.sjt_total + st.open_total + st.sr_total)

        summary = {
            "mean": sum(x.norm_score for x in out_scores)/len(out_scores) if out_scores else 0.0,
            "traps": float(traps),
            "consistency": cats["consistency_score"]/100.0,
            "synergy_boost": 0.0,
            "avg_rt_sr":  round(self.state.rt_sums.get('SR',0.0)/max(1,self.state.rt_counts.get('SR',0)),2) if self.state.rt_counts.get('SR',0)>0 else 0.0,
            "avg_rt_mcq": round(self.state.rt_sums.get('MCQ',0.0)/max(1,self.state.rt_counts.get('MCQ',0)),2) if self.state.rt_counts.get('MCQ',0)>0 else 0.0,
            "avg_rt_sjt": round(self.state.rt_sums.get('SJT',0.0)/max(1,self.state.rt_counts.get('SJT',0)),2) if self.state.rt_counts.get('SJT',0)>0 else 0.0,
            "oe_items": float(sum(self.state.domains[d].open_total for d in DOMAINS)),
            "oe_avg":   float(_safe_frac(sum(self.state.domains[d].open_sum for d in DOMAINS),
                                         sum(self.state.domains[d].open_total for d in DOMAINS))),
            "speed_score": cats["speed_score"],
            "precision_score": cats["precision_score"],
            "consistency_score": cats["consistency_score"],
            # echo baselines used
            "rt_baselines": self.base_rt,
            "total_items": int(total_items),
        }

        cap = CAP_SHORT if self.run_type == "short" else CAP_LONG
        obj_min = OBJ_MIN_SHORT if self.run_type == "short" else OBJ_MIN_LONG
        shortfalls = {
            d: max(0, obj_min - self.state.domains[d].obj_count)
            for d in DOMAINS
        }
        remaining_minima = sum(shortfalls.values())
        meta = {
            "run": self.run_type,
            "cap": cap,
            "steps_total": int(self._step),
            "steps_remaining": max(cap - self._step, 0),
        }
        if remaining_minima > 0:
            meta["incomplete"] = True
            meta["incomplete_reason"] = "CAP_BEFORE_MINIMA"
            meta["shortfalls"] = {d: v for d, v in shortfalls.items() if v > 0}
        else:
            meta["incomplete"] = False
            meta["shortfalls"] = {}

        summary["steps_total"] = int(self._step)
        summary["cap"] = cap
        summary["steps_remaining"] = max(cap - self._step, 0)

        # Write per-run item CSV
        run_id = os.getenv("RUN_ID", "")
        tag = f"{run_id}_{self.run_type}" if run_id else f"session_{int(time.time())}_{self.run_type}"
        self._write_items_csv(os.path.join("reports", f"{tag}.items.csv"))

        res = Result(run_type=self.run_type, domain_scores=out_scores, top_skills=top,
                     hidden_skills=hidden, traps_tripped=traps, consistency=summary["consistency"],
                     synergy_boost=0.0, unique_award=None, summary=summary)
        setattr(res, "meta", meta)
        res.audit_events = [
            {k: v for k, v in evt.items()}
            for evt in self.state.audit_events
        ]
        return res


def _dev_check_ladder() -> None:
    """Developer-only assertions for ladder progression and demotion guards."""

    short_seq = [(-1, 1), (-1, 1), (0, 1), (0, 1), (1, 1), (1, 0), (1, 1), (1, 1), (2, 1)]
    ds_short = DomainState(level=-1)
    se_values: List[float] = []
    print("[dev] short run synthetic progression:")
    for idx, (lvl, correct) in enumerate(short_seq, 1):
        info = _apply_objective_step(ds_short, lvl, bool(correct), "short", target_level=ds_short.level)
        se_values.append(ds_short.se)
        print(
            f"  step {idx:02d} lvl {info['level_before']:+d}->{info['level_after']:+d} "
            f"se={ds_short.se:.4f} window={info['window']}"
        )
    assert ds_short.level == 2, "short ladder should reach +2"
    assert ds_short.obj_count <= OBJ_MAX_SHORT, "short ladder exceeded objective cap"
    for prev, cur in zip(se_values, se_values[1:]):
        assert cur <= prev + 1e-6, "SE should not increase in short synthetic run"

    long_seq = [
        (-1, 1), (-1, 1), (-1, 1), (-1, 0),
        (0, 1), (0, 1), (0, 1), (0, 0),
        (1, 1), (1, 1), (1, 1), (1, 0),
        (2, 1),
    ]
    ds_long = DomainState(level=-1)
    for lvl, correct in long_seq:
        _apply_objective_step(ds_long, lvl, bool(correct), "long", target_level=ds_long.level)
    assert ds_long.level == 2, "long ladder should reach +2"
    assert ds_long.obj_count <= OBJ_MAX_LONG, "long ladder exceeded objective cap"

    ds_demo = DomainState(level=1)
    ds_demo.recent_window = []
    ds_demo.level_entry_seen = 0
    _apply_objective_step(ds_demo, 1, False, "short", target_level=1)
    assert ds_demo.level == 1, "first miss at new level should not demote"
    _apply_objective_step(ds_demo, 1, False, "short", target_level=1)
    _apply_objective_step(ds_demo, 1, False, "short", target_level=1)
    assert ds_demo.level == 0, "two misses in last three should demote"

    print("[dev] ladder checks passed (short and long sequences, demotion guard).")


def _dev_check_sr_behavior() -> None:
    domain = DOMAINS[0] if DOMAINS else "Analytical"
    base = DomainState(theta=0.25)
    score_a, _, _ = _composite(domain, base)
    heavier_sr = DomainState(theta=0.25, sr_sum=40.0, sr_total=40)
    score_b, _, _ = _composite(domain, heavier_sr)
    assert abs(score_a - score_b) < 1e-9, "SR weight should be zero in composite score"

    assert _sr_shift_from_mean(2.0) == SR_START_SHIFT.get("low", -1)
    assert _sr_shift_from_mean(3.5) == SR_START_SHIFT.get("mid", 0)
    assert _sr_shift_from_mean(4.8) == SR_START_SHIFT.get("high", 1)

    print("[dev] SR weight & start-level shift checks passed.")


if __name__ == "__main__":  # pragma: no cover - developer diagnostics only
    _dev_check_sr_behavior()
    _dev_check_ladder()
