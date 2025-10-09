# skill_core/policy.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import math
import random

from .question_bank import DOMAINS
from .config import (
    SE_TARGET_SHORT,
    SE_TARGET_LONG,
    OBJ_MIN_SHORT,
    OBJ_MAX_SHORT,
    OBJ_MIN_LONG,
    OBJ_MAX_LONG,
    LEVEL_MIN,
    LEVEL_MAX,
    CAP_SHORT,
    CAP_LONG,
    OPEN_ENABLED_LONG,
    OPEN_GATE1_MIN_OBJ,
    OPEN_GATE1_SE_MAX,
    OPEN_GATE2_SE_MAX,
    OPEN_GATE2_MIN_R1,
    OPEN_LEVELS,
    OPEN_Z_SHIFT,
    SR_PER_DOMAIN_SHORT,
    SR_PER_DOMAIN_LONG,
    DEBUG_SEED,
)


def _clamp_level(level: int) -> int:
    return max(LEVEL_MIN, min(LEVEL_MAX, int(level)))


@dataclass
class DomainHistory:
    asked_ids: List[str] = field(default_factory=list)
    sr_count: int = 0
    obj_count: int = 0
    open_count: int = 0
    obj_correct_frac: float = 0.0
    level: int = 0
    level_stats: Dict[int, Dict[str, int]] = field(default_factory=dict)
    recent_window: List[bool] = field(default_factory=list)
    level_entry_seen: int = 0
    obj_done: bool = False
    open_contrib: List[Dict[str, object]] = field(default_factory=list)
    obj_info_total: float = 0.0
    open_info_total: float = 0.0


@dataclass
class PolicyState:
    run_type: str
    theta: Dict[str, float]
    se: Dict[str, float]
    asked: Set[str]
    seen_variant_groups: Set[str]
    step: int
    hist: Dict[str, DomainHistory]
    info_history: List[float]
    mirrored_domains_planned: Set[str] = field(default_factory=set)
    last_domain_id: Optional[str] = None
    last_item_type: Optional[str] = None


class QuestionPolicy:
    """Difficulty-aware ladder policy with per-domain rotation guards."""

    def __init__(self, items: List, run_type: str):
        self.run_type = run_type if run_type in ("short", "long") else "long"
        self.items = items
        self._domain_index = {d: idx for idx, d in enumerate(DOMAINS)}
        seed = DEBUG_SEED
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self.rng = random.Random(int(seed))

        levels = range(LEVEL_MIN, LEVEL_MAX + 1)
        self._obj_index: Dict[str, Dict[int, List]] = {
            d: {lvl: [] for lvl in levels} for d in DOMAINS
        }
        self._sr_index: Dict[str, List] = {d: [] for d in DOMAINS}
        self._open_levels: Tuple[int, ...] = tuple(sorted({int(lvl) for lvl in OPEN_LEVELS}))
        self._open_index: Dict[str, Dict[int, List]] = {
            d: {lvl: [] for lvl in self._open_levels} for d in DOMAINS
        }
        self._last_pick_was_sr = False

        for it in items:
            domain = getattr(it, "domain", None)
            if domain not in self._obj_index:
                continue
            typ = getattr(it, "type", None)
            if typ in ("MCQ", "SJT"):
                lvl = _clamp_level(int(getattr(it, "difficulty", 0) or 0))
                self._obj_index[domain][lvl].append(it)
            elif typ == "SR":
                self._sr_index[domain].append(it)
            elif typ == "OPEN":
                raw_lvl = int(getattr(it, "difficulty", 0) or 0)
                lvl = max(self._open_levels[0], min(self._open_levels[-1], raw_lvl))
                if lvl in self._open_index[domain]:
                    self._open_index[domain][lvl].append(it)

    @staticmethod
    def _bounds(run_type: str) -> tuple[float, int, int, int]:
        if run_type == "short":
            return SE_TARGET_SHORT, OBJ_MIN_SHORT, OBJ_MAX_SHORT, SR_PER_DOMAIN_SHORT
        return SE_TARGET_LONG, OBJ_MIN_LONG, OBJ_MAX_LONG, SR_PER_DOMAIN_LONG

    @staticmethod
    def _cap_for(run_type: str) -> int:
        return CAP_SHORT if run_type == "short" else CAP_LONG

    def should_stop(self, st: PolicyState) -> bool:
        _, obj_min, obj_max, sr_required = self._bounds(st.run_type)
        cap = self._cap_for(st.run_type)

        if st.step >= cap:
            return True

        remaining_obj_minima = sum(
            max(0, obj_min - st.hist.get(d, DomainHistory()).obj_count)
            for d in DOMAINS
        )
        if remaining_obj_minima > 0:
            return False

        for d in DOMAINS:
            hist = st.hist.get(d, DomainHistory())
            if not hist.obj_done and hist.obj_count < obj_max:
                return False

        sr_needed = 0
        if sr_required > 0:
            sr_needed = sum(
                max(0, sr_required - st.hist.get(d, DomainHistory()).sr_count)
                for d in DOMAINS
            )
            if sr_needed > 0:
                if any(
                    st.hist.get(d, DomainHistory()).sr_count < sr_required
                    and self._sr_available(d, st)
                    for d in DOMAINS
                ):
                    return False
                return True

        if self.run_type == "long" and OPEN_ENABLED_LONG and sr_needed == 0:
            if self._open_candidate_exists(st, obj_min):
                return False

        return False

    def next_item(self, st: PolicyState):
        se_target, obj_min, obj_max, sr_required = self._bounds(st.run_type)
        cap = self._cap_for(st.run_type)
        steps_left = max(cap - st.step, 0)
        if steps_left <= 0:
            return None

        remaining_obj_minima = [
            d for d in DOMAINS if st.hist.get(d, DomainHistory()).obj_count < obj_min
        ]
        if remaining_obj_minima:
            domains = self._sort_candidates(remaining_obj_minima, st)
            item = self._serve_objective(domains, st)
            if item:
                self._last_pick_was_sr = False
            return item

        stage_b = [
            d
            for d in DOMAINS
            if not st.hist.get(d, DomainHistory()).obj_done
            and st.se.get(d, 0.0) > se_target
        ]
        stage_c = [
            d
            for d in DOMAINS
            if not st.hist.get(d, DomainHistory()).obj_done
            and st.hist.get(d, DomainHistory()).obj_count < obj_max
        ]

        sr_needed = 0
        if sr_required > 0:
            sr_needed = sum(
                max(0, sr_required - st.hist.get(d, DomainHistory()).sr_count)
                for d in DOMAINS
            )

        sr_only = sr_needed > 0 and steps_left <= sr_needed
        if sr_needed > 0:
            sr_domain = self._sr_domain(st, obj_min, sr_required)
            if sr_domain is None and sr_only:
                return None
            if sr_domain is not None and (
                sr_only or (not self._last_pick_was_sr or (not stage_b and not stage_c))
            ):
                item = self._pick_sr_item(sr_domain, st)
                if item:
                    setattr(item, "_target_level", None)
                    setattr(item, "_served_level", None)
                    self._last_pick_was_sr = True
                    return item

        open_item = self._maybe_serve_open(st, obj_min, sr_needed)
        if open_item is not None:
            self._last_pick_was_sr = False
            return open_item

        if stage_b:
            domains = self._sort_candidates(stage_b, st)
            item = self._serve_objective(domains, st)
            if item:
                self._last_pick_was_sr = False
                return item

        if stage_c:
            domains = self._sort_candidates(stage_c, st)
            item = self._serve_objective(domains, st)
            if item:
                self._last_pick_was_sr = False
                return item

        item = self._fallback_objective(st)
        self._last_pick_was_sr = False
        return item

    def _sort_candidates(self, domains: List[str], st: PolicyState) -> List[str]:
        def key(dom: str) -> tuple:
            se_val = st.se.get(dom, 0.0)
            obj_count = st.hist.get(dom, DomainHistory()).obj_count
            rotation = 0 if st.last_domain_id == dom else -1
            return (-se_val, obj_count, rotation, self._domain_index[dom])

        return sorted(domains, key=key)

    def _level_order(self, base_level: int) -> List[int]:
        order: List[int] = []
        for delta in (0, 1, -1, 2, -2):
            lvl = _clamp_level(base_level + delta)
            if lvl not in order:
                order.append(lvl)
        return order

    def _open_level_order(self, preferred: int) -> List[int]:
        def key(lvl: int) -> Tuple[float, int]:
            return (abs(lvl - preferred), -lvl)

        return sorted(self._open_levels, key=key)

    def _pick_open_level(self, theta_val: float) -> int:
        return min(self._open_levels, key=lambda lvl: abs(theta_val - lvl))

    def _select_open_item(
        self, domain: str, preferred_level: int, st: PolicyState, *, peek: bool = False
    ) -> Tuple[Optional[object], Optional[int]]:
        for lvl in self._open_level_order(preferred_level):
            pool = self._open_index.get(domain, {}).get(lvl, [])
            options = [it for it in pool if self._is_unseen(it, st)]
            if options:
                item = options[0] if peek else self.rng.choice(options)
                if not peek:
                    setattr(item, "_open_target_level", preferred_level)
                    setattr(item, "_open_served_level", lvl)
                return item, lvl
        return None, None

    def _open_candidates(
        self, st: PolicyState, obj_min: int
    ) -> List[Tuple[str, object, float, int, int, int]]:
        if self.run_type != "long" or not OPEN_ENABLED_LONG:
            return []

        if any(st.hist.get(d, DomainHistory()).obj_count < obj_min for d in DOMAINS):
            return []

        if st.last_item_type not in ("MCQ", "SJT"):
            return []

        candidates: List[Tuple[str, object, float, int, int, int]] = []
        for domain in DOMAINS:
            hist = st.hist.get(domain, DomainHistory())
            if hist.open_count >= 2:
                continue

            se_val = st.se.get(domain, 1.0)
            theta_val = st.theta.get(domain, 0.0)
            target_level: Optional[int] = None

            if hist.open_count == 0:
                if hist.obj_count < OPEN_GATE1_MIN_OBJ:
                    continue
                if se_val > OPEN_GATE1_SE_MAX:
                    continue
                target_level = self._pick_open_level(theta_val)
            elif hist.open_count == 1:
                if hist.obj_count < OPEN_GATE1_MIN_OBJ:
                    continue
                if se_val > OPEN_GATE2_SE_MAX:
                    continue
                if not hist.open_contrib:
                    continue
                first = hist.open_contrib[0]
                if first.get("ignored") is not None:
                    continue
                r1 = float(first.get("r", 0.0))
                if r1 < OPEN_GATE2_MIN_R1:
                    continue
                b1 = int(first.get("b", 0))
                mu1 = float(first.get("mu", 0.5))
                denom = max(mu1 * (1.0 - mu1), 1e-6)
                z1 = (r1 - mu1) / math.sqrt(denom)
                target_level = b1
                recent = hist.recent_window[-4:]
                last_two = recent[-2:] if len(recent) >= 2 else []
                has_miss = any(not x for x in hist.recent_window[-3:])
                if len(last_two) == 2 and all(last_two) and z1 >= OPEN_Z_SHIFT:
                    target_level = min(b1 + 1, self._open_levels[-1])
                elif has_miss and z1 <= -OPEN_Z_SHIFT:
                    target_level = max(b1 - 1, self._open_levels[0])
            else:
                continue

            if target_level is None:
                continue

            item, served_level = self._select_open_item(
                domain, int(target_level), st, peek=True
            )
            if item is None:
                continue

            candidates.append(
                (
                    domain,
                    item,
                    se_val,
                    hist.open_count,
                    int(target_level),
                    int(served_level if served_level is not None else target_level),
                )
            )

        return candidates

    def _open_candidate_exists(self, st: PolicyState, obj_min: int) -> bool:
        return bool(self._open_candidates(st, obj_min))

    def _maybe_serve_open(
        self, st: PolicyState, obj_min: int, sr_needed: int
    ) -> Optional[object]:
        if sr_needed > 0:
            return None

        candidates = self._open_candidates(st, obj_min)
        if not candidates:
            return None

        def key(entry: Tuple[str, object, float, int, int, int]) -> Tuple:
            dom, _, se_val, open_count, _, _ = entry
            rotation = 0 if st.last_domain_id == dom else -1
            return (-se_val, open_count, rotation, self._domain_index[dom])

        domain, _, _, _, target_level, _ = sorted(candidates, key=key)[0]
        item, served_level = self._select_open_item(domain, target_level, st)
        if item is None:
            return None
        if served_level is None:
            served_level = target_level
        setattr(item, "_open_target_level", target_level)
        setattr(item, "_open_served_level", served_level)
        return item

    def _is_unseen(self, item, st: PolicyState) -> bool:
        iid = getattr(item, "id", "")
        if iid in st.asked:
            return False
        vg = getattr(item, "variant_group", None)
        if vg and vg in st.seen_variant_groups:
            return False
        return True

    def _objective_item(self, domain: str, st: PolicyState) -> Optional:
        base_level = st.hist.get(domain, DomainHistory()).level
        for lvl in self._level_order(base_level):
            pool = self._obj_index.get(domain, {}).get(lvl, [])
            options = [it for it in pool if self._is_unseen(it, st)]
            if options:
                item = self.rng.choice(options)
                setattr(item, "_target_level", base_level)
                setattr(item, "_served_level", lvl)
                return item
        return None

    def _serve_objective(self, domains: List[str], st: PolicyState):
        for dom in domains:
            item = self._objective_item(dom, st)
            if item is not None:
                return item
        return None

    def _fallback_objective(self, st: PolicyState):
        for dom in DOMAINS:
            for lvl in range(LEVEL_MIN, LEVEL_MAX + 1):
                pool = self._obj_index.get(dom, {}).get(lvl, [])
                options = [it for it in pool if self._is_unseen(it, st)]
                if options:
                    item = self.rng.choice(options)
                    setattr(item, "_target_level", st.hist.get(dom, DomainHistory()).level)
                    setattr(item, "_served_level", lvl)
                    return item
        return None

    def _sr_available(self, domain: str, st: PolicyState) -> bool:
        for it in self._sr_index.get(domain, []):
            if self._is_unseen(it, st):
                return True
        return False

    def _sr_domain(self, st: PolicyState, obj_min: int, sr_required: int) -> Optional[str]:
        domains = [
            d
            for d in DOMAINS
            if st.hist.get(d, DomainHistory()).obj_count >= obj_min
            and st.hist.get(d, DomainHistory()).sr_count < sr_required
            and self._sr_available(d, st)
        ]
        if not domains:
            return None

        def key(dom: str) -> tuple:
            sr_count = st.hist.get(dom, DomainHistory()).sr_count
            rotation = 0 if st.last_domain_id == dom else -1
            return (sr_count, rotation, self._domain_index[dom])

        return sorted(domains, key=key)[0]

    def _pick_sr_item(self, domain: str, st: PolicyState):
        pool = [it for it in self._sr_index.get(domain, []) if self._is_unseen(it, st)]
        if not pool:
            return None
        return self.rng.choice(pool)


def _simulate_run(run_type: str, pattern: List[Tuple[int, int]]) -> List[int]:
    """Developer helper: returns level trajectory for synthetic answers."""

    from .engine import DomainState, _apply_objective_step

    ds = DomainState(level=-1)
    levels = [ds.level]
    for level, correct in pattern:
        _apply_objective_step(ds, level, bool(correct), run_type, target_level=ds.level)
        levels.append(ds.level)
    return levels


def _dev_check_sr_scheduling() -> None:
    from .types import Item

    items: List[Item] = []
    for dom in DOMAINS:
        for lvl in range(LEVEL_MIN, LEVEL_MAX + 1):
            items.append(
                Item(
                    id=f"{dom}_mcq_{lvl:+d}",
                    domain=dom,
                    type="MCQ",
                    text="stub",
                    options=["A", "B"],
                    correct=0,
                    difficulty=lvl,
                )
            )
        for idx in range(SR_PER_DOMAIN_LONG):
            items.append(
                Item(
                    id=f"{dom}_sr_{idx}",
                    domain=dom,
                    type="SR",
                    text="stub",
                    options=["1", "2", "3", "4", "5"],
                )
            )

    policy = QuestionPolicy(items, "long")
    obj_min = OBJ_MIN_LONG
    sr_required = SR_PER_DOMAIN_LONG

    hist = {dom: DomainHistory(obj_count=obj_min - 1, sr_count=0) for dom in DOMAINS}
    base_state = PolicyState(
        run_type="long",
        theta={dom: 0.0 for dom in DOMAINS},
        se={dom: 1.0 for dom in DOMAINS},
        asked=set(),
        seen_variant_groups=set(),
        step=0,
        hist=hist,
        info_history=[],
        mirrored_domains_planned=set(),
        last_domain_id=None,
        last_item_type="MCQ",
    )

    assert (
        policy._sr_domain(base_state, obj_min, sr_required) is None
    ), "SR should not schedule before objective minima"

    first_dom = DOMAINS[0]
    hist[first_dom].obj_count = obj_min
    sr_dom = policy._sr_domain(base_state, obj_min, sr_required)
    assert sr_dom == first_dom, "SR should unlock once minima met"

    hist[first_dom].sr_count = sr_required
    assert (
        policy._sr_domain(base_state, obj_min, sr_required) is None
    ), "SR should respect per-domain cap"

    print("[dev] SR scheduling respects objective minima and per-domain caps.")


if __name__ == "__main__":
    short_path = [
        (-1, 1), (-1, 1),
        (0, 1), (0, 1),
        (1, 1), (1, 0), (1, 1), (1, 1),
        (2, 1),
    ]
    long_path = [
        (-1, 1), (-1, 1), (-1, 1), (-1, 0),
        (0, 1), (0, 1), (0, 1), (0, 0),
        (1, 1), (1, 1), (1, 1), (1, 0),
        (2, 1),
    ]
    _dev_check_sr_scheduling()
    print("Short ladder levels:", _simulate_run("short", short_path))
    print("Long ladder levels:", _simulate_run("long", long_path))
