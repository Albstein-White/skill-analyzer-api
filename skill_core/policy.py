# skill_core/policy.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
import os, random
from .question_bank import DOMAINS

@dataclass
class DomainHistory:
    asked_ids: List[str] = field(default_factory=list)
    sr_count:   int = 0
    obj_count:  int = 0     # MCQ + SJT
    open_count: int = 0
    obj_correct_frac: float = 0.0  # 0..1 objective accuracy for gating

@dataclass
class PolicyState:
    run_type: str
    theta: Dict[str, float]
    se: Dict[str, float]
    asked: Set[str]
    seen_variants: Set[str]
    step: int
    hist: Dict[str, DomainHistory]
    info_history: List[float]
    mirrored_domains_planned: Set[str] = field(default_factory=set)

class QuestionPolicy:
    """
    Count-based policy with OPEN eligibility.
    Short: no OPEN by default (cap handles ceiling). Enable via SHORT_ALLOW_OPEN=1 to include at most 1 OPEN/domain when eligible.
    Long: OBJ and SR quotas, OPEN required when eligible. On autoplay 'perfect', force ≥1 OPEN/domain.
    """

    def __init__(self, items: List, run_type: str):
        self.run_type = run_type if run_type in ("short", "long") else "long"
        self.items = items
        idx: Dict[str, Dict[str, List]] = {d: {"MCQ": [], "SJT": [], "OPEN": [], "SR": []} for d in DOMAINS}
        for it in items:
            d = getattr(it, "domain", None)
            t = getattr(it, "type", None)
            if d in idx and t in idx[d]:
                idx[d][t].append(it)
        self.idx = idx

        self.max_steps = 56 if self.run_type == "short" else 120
        self.MIN_OBJ_FOR_OPEN = 2 if self.run_type == "short" else 4
        self.MIN_OBJ_ACC_FOR_OPEN = 0.60

        self.short_allow_open = os.getenv("SHORT_ALLOW_OPEN", "0") == "1"
        self.force_open_each_domain = (os.getenv("PROFILE", "").lower() == "perfect" and self.run_type == "long")

    def _quotas(self) -> Dict[str, Dict[str, int]]:
        if self.run_type == "short":
            # default: no OPENs in short
            open_q = 1 if self.short_allow_open else 0
            return {d: {"OBJ": 4, "OPEN": open_q, "SR": 2} for d in DOMAINS}
        # long
        return {d: {"OBJ": 8, "OPEN": 2, "SR": 2} for d in DOMAINS}

    def _counts(self, st: PolicyState, d: str) -> Dict[str, int]:
        h = st.hist.get(d, DomainHistory())
        return {"OBJ": int(h.obj_count), "OPEN": int(h.open_count), "SR": int(h.sr_count)}

    def _eligible_for_open(self, st: PolicyState, d: str) -> bool:
        if not self.idx.get(d, {}).get("OPEN"):
            return False
        if self.force_open_each_domain:
            return True
        h = st.hist.get(d, DomainHistory())
        return (h.obj_count >= self.MIN_OBJ_FOR_OPEN) and (h.obj_correct_frac >= self.MIN_OBJ_ACC_FOR_OPEN)

    def should_stop(self, st: PolicyState) -> bool:
        if st.step >= self.max_steps:
            return True
        quotas = self._quotas()
        for d in DOMAINS:
            q = quotas[d]; c = self._counts(st, d)
            if c["OBJ"] < q["OBJ"] or c["SR"] < q["SR"]:
                return False
            # OPEN only matters if quota>0 and eligible
            if q["OPEN"] > 0 and self._eligible_for_open(st, d):
                if c["OPEN"] < q["OPEN"]:
                    return False
        return True

    def _domain_deficit(self, st: PolicyState, d: str) -> Dict[str, int]:
        q = self._quotas()[d]; c = self._counts(st, d)
        def_map = {k: max(0, q[k] - c[k]) for k in ("OBJ", "OPEN", "SR")}
        if q["OPEN"] == 0 or not self._eligible_for_open(st, d):
            def_map["OPEN"] = 0
        return def_map

    def _pick_domain(self, st: PolicyState) -> Optional[str]:
        best_d, best_key = None, (-1, -1)
        for d in DOMAINS:
            h = st.hist.get(d, DomainHistory())
            def_map = self._domain_deficit(st, d)
            total_def = def_map["OBJ"] + def_map["OPEN"] + def_map["SR"]
            key = (total_def, -len(h.asked_ids))
            if key > best_key:
                best_d, best_key = d, key
        return best_d

    def _pick_type(self, st: PolicyState, d: str) -> str:
        def_map = self._domain_deficit(st, d)
        # prefer OBJ > OPEN > SR
        order = ["OBJ", "OPEN", "SR"]
        order.sort(key=lambda k: (-def_map[k], {"OBJ":0, "OPEN":1, "SR":2}[k]))
        choice = order[0]
        if choice == "OPEN" and (self._quotas()[d]["OPEN"] == 0 or not self._eligible_for_open(st, d)):
            choice = "OBJ" if def_map["OBJ"] > 0 else "SR"
        return choice

    def _candidates(self, d: str, typ: str) -> List:
        if typ == "OBJ":
            return (self.idx[d]["MCQ"] or []) + (self.idx[d]["SJT"] or [])
        return self.idx[d][typ]

    def next_item(self, st: PolicyState):
        d = self._pick_domain(st)
        if d is None:
            return None
        typ_choice = self._pick_type(st, d)
        pool = self._candidates(d, typ_choice)

        def unseen(it) -> bool:
            iid = getattr(it, "id", "")
            vg = getattr(it, "variant_group", None)
            if iid in st.asked: return False
            if vg and vg in st.seen_variants: return False
            return True

        cand = [it for it in pool if unseen(it)]
        if not cand:
            any_dom = [it for it in (self.idx[d]["MCQ"] + self.idx[d]["SJT"] + self.idx[d]["OPEN"] + self.idx[d]["SR"]) if unseen(it)]
            if not any_dom:
                all_unseen = [it for it in self.items if getattr(it, "id", "") not in st.asked]
                return random.choice(all_unseen) if all_unseen else None
            return random.choice(any_dom)
        return random.choice(cand)
