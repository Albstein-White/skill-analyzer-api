from __future__ import annotations

import json
import logging
from typing import List

from .config import DEBUG_SEED, TRACE_FIELDS, DEBUG_TRACE, STAGING_PROFILE
from .engine import AdaptiveSession
from .plan import generate_plan
from .question_bank import DOMAINS
from .types import Answer, Item


def _maybe_enable_trace() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if DEBUG_TRACE:
        logging.getLogger("skill_core.engine").setLevel(logging.INFO)


def _synthetic_bank() -> List[Item]:
    items: List[Item] = []
    for domain in DOMAINS:
        for level in range(-2, 3):
            for idx in range(3):
                items.append(
                    Item(
                        id=f"smoke_{domain}_mcq_{level}_{idx}",
                        domain=domain,
                        type="MCQ",
                        text=f"{domain} MCQ {level}",
                        options=["A", "B", "C", "D"],
                        correct=0,
                        difficulty=level,
                        variant_group=f"smoke_{domain}_mcq_{level}_{idx}",
                    )
                )
        for idx in range(2):
            items.append(
                Item(
                    id=f"smoke_{domain}_sr_{idx}",
                    domain=domain,
                    type="SR",
                    text=f"Rate your {domain} confidence",
                    options=["0", "1", "2", "3", "4"],
                    difficulty=0,
                    variant_group=f"smoke_{domain}_sr_{idx}",
                )
            )
        for level in (-1, 0, 1):
            items.append(
                Item(
                    id=f"smoke_{domain}_open_{level}",
                    domain=domain,
                    type="OPEN",
                    text=f"Describe a {domain} challenge",
                    difficulty=level,
                    variant_group=f"smoke_{domain}_open_{level}",
                )
            )
    return items


def _auto_answer(item: Item) -> Answer:
    if item.type == "MCQ":
        return Answer(item_id=item.id, value=item.correct or 0, rt_sec=18.0)
    if item.type == "SJT":
        return Answer(item_id=item.id, value=0, rt_sec=22.0)
    if item.type == "SR":
        return Answer(item_id=item.id, value=3, rt_sec=5.0)
    words = "practice".split()
    open_text = " ".join(words * 80)
    return Answer(item_id=item.id, value=open_text, rt_sec=120.0)


def _trace_fields() -> str:
    return ", ".join(TRACE_FIELDS)


def run_smoke_session() -> None:
    _maybe_enable_trace()

    from skill_core import engine as eng

    bank = _synthetic_bank()
    eng.load_bank = lambda: list(bank)

    session = AdaptiveSession("long")
    logging.info("Starting synthetic long run with DEBUG_SEED=%s", DEBUG_SEED)
    logging.info("Trace fields: %s", _trace_fields())

    while True:
        item = session.next_item()
        if item is None:
            break
        answer = _auto_answer(item)
        session.answer_current(answer)

    result = session.finalize()
    payload = json.loads(json.dumps(result, default=lambda o: getattr(o, "__dict__", o)))
    cfg = {}
    plan = generate_plan(payload, cfg)

    if STAGING_PROFILE:
        meta = payload.get("meta", {}) or {}
        shortfalls = meta.get("shortfalls") or {}
        for domain in DOMAINS:
            state = session.state.domains.get(domain)
            theta = getattr(state, "theta", 0.0)
            se = getattr(state, "se", 0.0)
            b_stable = getattr(state, "b_stable", 0)
            obj_count = getattr(state, "obj_count", 0)
            open_count = getattr(state, "open_count", 0)
            print(f"{domain}: θ={theta:.2f} SE={se:.2f} b_stable={b_stable:+d} OBJ={obj_count} OPEN={open_count}")
        steps_total = meta.get("steps_total", 0)
        cap_limit = meta.get("cap", 0)
        total_sr = sum(getattr(st, "sr_count", 0) for st in session.state.domains.values())
        total_open = sum(getattr(st, "open_count", 0) for st in session.state.domains.values())
        minima_unmet = sum(shortfalls.values()) > 0 if isinstance(shortfalls, dict) else False
        print(
            f"cap={steps_total}/{cap_limit} minima_unmet={minima_unmet} sr_used={total_sr} open_used={total_open}"
        )
        return

    logging.info(
        "Run complete: steps=%s cap=%s",
        payload.get("meta", {}).get("steps_total"),
        payload.get("meta", {}).get("cap"),
    )

    for domain_score in payload.get("domain_scores", []):
        logging.info(
            "Domain %s: theta=%.3f se=%.3f level=%s stable=%s obj=%s open=%s",
            domain_score.get("domain"),
            float(domain_score.get("theta", 0.0)),
            float(domain_score.get("se", 0.0)),
            domain_score.get("level"),
            domain_score.get("b_stable"),
            domain_score.get("obj_count"),
            domain_score.get("open_count"),
        )
        logging.info(
            "  items_by_level=%s",
            domain_score.get("items_by_level"),
        )
        contribs = domain_score.get("open_contrib") or []
        for entry in contribs:
            logging.info(
                "  OPEN @%s r=%.2f mu=%.2f dtheta=%.3f%s",
                entry.get("b"),
                float(entry.get("r", 0.0)),
                float(entry.get("mu", 0.0)),
                float(entry.get("dtheta", 0.0)),
                " (scaled)" if entry.get("scaled") else "",
            )

    logging.info("Plan entries generated: %d", len(plan))


if __name__ == "__main__":  # pragma: no cover
    run_smoke_session()

