# skill_core/engine.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math, random

from .types import Item, Answer, DomainScore, HiddenSkill, Result
from .question_bank import load_bank, DOMAINS
from .scoring import score_item
from .validators import count_traps, consistency_index
from .rarity import tier, rarity_label
from .config import load_config, seed_rng
from .policy import QuestionPolicy, PolicyState, DomainHistory

# Composite base weights (domain score)
W_OPEN_BASE = 0.60
W_OBJ       = 0.35
W_SR        = 0.05

# Baseline response times in seconds for Speed category (summary-level only)
BASE_RT = {"MCQ": 20.0, "SJT": 25.0, "SR": 10.0, "OPEN": 90.0}

@dataclass
class DomainState:
    # Objective
    mcq_correct: int = 0
    mcq_total:   int = 0
    sjt_sum:     float = 0.0   # sum of [0..1] credits
    sjt_total:   int = 0
    # OPEN
    open_sum:    float = 0.0
    open_total:  int = 0
    # SR
    sr_sum:      float = 0.0
    sr_total:    int = 0
    # misc
    max_diff: int = -3

@dataclass
class EngineState:
    domains: Dict[str, DomainState] = field(default_factory=lambda: {d: DomainState() for d in DOMAINS})
    answers: Dict[str, Answer] = field(default_factory=dict)
    rt_sums: Dict[str, float] = field(default_factory=lambda: {"SR": 0.0, "MCQ": 0.0, "SJT": 0.0, "OPEN": 0.0})
    rt_counts: Dict[str, int] = field(default_factory=lambda: {"SR": 0, "MCQ": 0, "SJT": 0, "OPEN": 0})

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

def _composite(st: DomainState) -> tuple[float, float, dict]:
    """
    Returns (score_0_100, se_approx, parts_dict).
    OPEN contributes weight only if present; its weight is gated by objective accuracy.
    """
    obj = _obj_frac(st)    # 0..1
    opn = _open_frac(st)   # 0..1
    sr  = _sr_frac(st)     # 0..1

    present = {"obj": (st.mcq_total + st.sjt_total) > 0, "open": st.open_total > 0, "sr": st.sr_total > 0}

    w_open_eff = (W_OPEN_BASE * (0.5 + 0.5 * obj)) if present["open"] else 0.0
    w_obj = W_OBJ if present["obj"] else 0.0
    w_sr  = W_SR  if present["sr"]  else 0.0
    w_sum = w_open_eff + w_obj + w_sr
    if w_sum <= 0:
        return 0.0, 0.60, {"obj": 0.0, "open": 0.0, "sr": 0.0}

    w_open = w_open_eff / w_sum
    w_obj  = w_obj       / w_sum
    w_sr   = w_sr        / w_sum

    score = 100.0 * (w_obj * obj + w_open * opn + w_sr * sr)

    # SE from component variances (Bernoulli approx), only for present parts
    var_obj  = (obj * (1 - obj)) / max(1, (st.mcq_total + st.sjt_total)) if present["obj"] else 0.0
    var_open = (opn * (1 - opn)) / max(1, st.open_total) if present["open"] else 0.0
    var_sr   = (sr  * (1 - sr )) / max(1, st.sr_total)  if present["sr"]  else 0.0
    var_total = (w_obj**2)*var_obj + (w_open**2)*var_open + (w_sr**2)*var_sr
    se = max(0.20, min(0.60, 0.20 + math.sqrt(max(0.0, var_total))))

    parts = {"obj": round(100.0*obj,1), "open": round(100.0*opn,1), "sr": round(100.0*sr,1)}
    return score, se, parts

class AdaptiveSession:
    def __init__(self, run_type: str):
        assert run_type in ("short","long")
        self.cfg = load_config(); seed_rng(self.cfg)
        self.run_type = run_type
        self.state = EngineState()
        self.items: list[Item] = load_bank()
        random.shuffle(self.items)

        self.asked: set[str] = set()
        self.seen_variants: set[str] = set()
        self._current: Optional[Item] = None
        self._step = 0

        self.policy = QuestionPolicy(self.items, run_type)
        self._id_to_item: Dict[str, Item] = {it.id: it for it in self.items}
        self._info_hist: List[float] = []

    def _policy_state(self) -> PolicyState:
        hist: Dict[str, DomainHistory] = {}
        for d in DOMAINS:
            dh = DomainHistory()
            dh.asked_ids = [iid for iid in self.asked if self._id_to_item[iid].domain == d]
            st = self.state.domains[d]
            dh.sr_count   = st.sr_total
            dh.obj_count  = st.mcq_total + st.sjt_total
            dh.open_count = st.open_total
            denom = float(st.mcq_total + st.sjt_total)
            dh.obj_correct_frac = (st.mcq_correct + st.sjt_sum) / denom if denom > 0 else 0.0
            hist[d] = dh
        return PolicyState(
            run_type=self.run_type, theta={}, se={}, asked=set(self.asked),
            seen_variants=set(self.seen_variants), step=self._step, hist=hist,
            info_history=self._info_hist, mirrored_domains_planned=getattr(self, "_mir_planned", set())
        )

    def next_item(self) -> Optional[Item]:
        st = self._policy_state()
        if self.policy.should_stop(st): return None
        it = self.policy.next_item(st)
        self._current = it
        return it

    def answer_current(self, answer: Answer) -> None:
        if not self._current: return
        it = self._current
        credit, _ = score_item(it, answer)

        ds = self.state.domains[it.domain]
        if answer.rt_sec is not None:
            self.state.rt_sums[it.type] += answer.rt_sec
            self.state.rt_counts[it.type] += 1

        if it.type == "MCQ":
            ds.mcq_total += 1
            ds.mcq_correct += 1 if credit >= 0.999 else 0
            ds.max_diff = max(ds.max_diff, int(getattr(it, "difficulty", 0) or 0))
        elif it.type == "SJT":
            ds.sjt_total += 1
            ds.sjt_sum   += float(credit)
        elif it.type == "OPEN":
            ds.open_total += 1
            ds.open_sum   += float(credit)
        elif it.type == "SR":
            ds.sr_total += 1
            ds.sr_sum   += float(credit)

        self.asked.add(it.id)
        if getattr(it, "variant_group", None):
            self.seen_variants.add(it.variant_group)
        self.state.answers[it.id] = answer
        self._step += 1
        self._current = None

    def _summary_categories(self) -> Dict[str, float]:
        """
        Returns Speed, Precision, Consistency in 0..100.
        No penalties applied to domain scores; these are separate.
        """
        # Objective accuracy across all domains (precision)
        obj_nums = []; obj_dens = []
        for d in DOMAINS:
            st = self.state.domains[d]
            n = (st.mcq_total + st.sjt_total)
            if n > 0:
                obj_nums.append(st.mcq_correct + st.sjt_sum)
                obj_dens.append(n)
        obj_acc = _safe_frac(sum(obj_nums), sum(obj_dens)) if obj_dens else 0.0
        precision = round(100.0 * obj_acc, 1)

        # Speed: normalize RT against baselines over available types, blend with accuracy to avoid gaming
        comps = []
        for t in ("MCQ", "SJT", "SR", "OPEN"):
            c = self.state.rt_counts.get(t, 0)
            if c <= 0: continue
            avg_rt = self.state.rt_sums.get(t, 0.0) / max(1, c)
            base = BASE_RT.get(t, 20.0)
            ratio = base / max(1e-6, avg_rt)  # >1 means faster than baseline
            ratio = min(max(ratio, 0.0), 1.5) # clamp
            comps.append(ratio / 1.5)         # map to 0..1
        speed_component = sum(comps) / len(comps) if comps else 0.0
        speed = round(100.0 * (0.6 * speed_component + 0.4 * obj_acc), 1)

        # Consistency: map validator index (assumed 0..1) to 0..100
        # If your consistency_index returns higher=better, use as-is; else invert here.
        cons = consistency_index(
            [it for it in self._id_to_item.values() if it.id in self.asked],
            self.state.answers
        )
        try:
            cons_val = float(cons)
        except Exception:
            cons_val = 0.0
        cons_val = min(max(cons_val, 0.0), 1.0)
        consistency = round(100.0 * cons_val, 1)

        return {"speed_score": speed, "precision_score": precision, "consistency_score": consistency}

    def finalize(self) -> Result:
        asked = [self._id_to_item[iid] for iid in self.asked if iid in self._id_to_item]
        traps = count_traps(asked, self.state.answers)

        out_scores: List[DomainScore] = []
        for d in DOMAINS:
            st = self.state.domains[d]
            score, se, parts = _composite(st)

            # A-tier cap (≤80) when no OPEN was answered.
            cap_applied = None
            if st.open_total == 0 and score > 80.0:
                score = 80.0
                cap_applied = "A"

            ds = DomainScore(
                domain=d,
                theta=0.0,                 # legacy unused
                se=se,
                norm_score=round(score, 1),
                tier=tier(score),
                rarity=rarity_label(score),
            )

            # counts + diagnostics for report
            asked_obj = int(st.mcq_total + st.sjt_total)
            asked_open = int(st.open_total)
            setattr(ds, "asked_obj", asked_obj)
            setattr(ds, "asked_open", asked_open)
            setattr(ds, "n", asked_obj + asked_open)
            setattr(ds, "obj_pct", parts["obj"])
            setattr(ds, "open_pct", parts["open"])
            setattr(ds, "sr_pct", parts["sr"])
            if cap_applied:
                setattr(ds, "cap", cap_applied)
            out_scores.append(ds)

        top = [x.domain for x in sorted(out_scores, key=lambda x: x.norm_score, reverse=True)[:5]]

        # summary categories
        cats = self._summary_categories()

        # undervalued skills (SR << objective) — up to 5
        hidden: List[HiddenSkill] = []
        for d, st in self.state.domains.items():
            obj = _obj_frac(st); sr = _sr_frac(st)
            gap = obj - sr
            if (st.mcq_total + st.sjt_total) >= 3 and st.sr_total >= 3 and gap >= 0.12:
                conf = "High" if gap>=0.20 else ("Medium" if gap>=0.16 else "Low")
                hidden.append(HiddenSkill(domain=d, confidence=conf, reason=f"Objective−SR gap {gap:.2f}"))
        hidden = sorted(hidden, key=lambda h: float(h.reason.split()[-1]), reverse=True)[:5] if hidden else []

        summary = {
            "mean": sum(x.norm_score for x in out_scores)/len(out_scores) if out_scores else 0.0,
            "traps": float(traps),
            "consistency": cats["consistency_score"]/100.0,  # keep legacy field as 0..1
            "synergy_boost": 0.0,
            "avg_rt_sr":  round(self.state.rt_sums.get('SR',0.0)/max(1,self.state.rt_counts.get('SR',0)),2) if self.state.rt_counts.get('SR',0)>0 else 0.0,
            "avg_rt_mcq": round(self.state.rt_sums.get('MCQ',0.0)/max(1,self.state.rt_counts.get('MCQ',0)),2) if self.state.rt_counts.get('MCQ',0)>0 else 0.0,
            "avg_rt_sjt": round(self.state.rt_sums.get('SJT',0.0)/max(1,self.state.rt_counts.get('SJT',0)),2) if self.state.rt_counts.get('SJT',0)>0 else 0.0,
            "oe_items": float(sum(self.state.domains[d].open_total for d in DOMAINS)),
            "oe_avg":   float(_safe_frac(sum(self.state.domains[d].open_sum for d in DOMAINS),
                                         sum(self.state.domains[d].open_total for d in DOMAINS))),
            # new categories (0..100)
            "speed_score": cats["speed_score"],
            "precision_score": cats["precision_score"],
            "consistency_score": cats["consistency_score"],
        }

        return Result(
            run_type=self.run_type,
            domain_scores=out_scores,
            top_skills=top,
            hidden_skills=hidden,             # shown as "Undervalued skills" in the report
            traps_tripped=traps,
            consistency=summary["consistency"],# legacy
            synergy_boost=0.0,
            unique_award=None,
            summary=summary,
        )

# Simple evaluators (legacy)
def evaluate(answers, run_type: str):
    sess = AdaptiveSession(run_type=run_type)
    for a in answers:
        sess.answer_current(a)
    return sess.finalize()

def evaluate_short(answers): return evaluate(answers, "short")
def evaluate_long(answers):  return evaluate(answers, "long")
