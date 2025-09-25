from __future__ import annotations
import json, itertools
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import skill_core.question_bank as qb
import skill_core as sc

TARGETS = {"MCQ_min": 12, "SJT_min": 6, "OPEN_min": 2, "MCQ_plus2_min": 2}
DOMAINS = ["Analytical","Mathematical","Verbal","Memory","Spatial","Creativity","Strategy","Social"]

MCQ_TEMPLATES = {
    "Analytical": ("Best first step when a model underperforms validation?", ["Check data leakage","Add more features","Tune random seed","Increase batch size"], 0),
    "Mathematical": ("Solve 2x+6=14; x=?", ["4","5","6","3"], 0),
    "Verbal": ("Choose the clearest sentence.", ["We will use data to improve","An improvement will be achieved by utilization","We will utilize for the purpose of betterment","Resources will be utilized to be improved"], 0),
    "Memory": ("Which method best supports long-term retention?", ["Spaced retrieval","Single reread","Highlighting only","Skimming"], 0),
    "Spatial": ("Rotate (x,y) by 90° CCW.", ["-y, x","y, -x","x, y","-x, -y"], 0),
    "Creativity": ("Which tends to increase idea originality?", ["Specific constraints","No constraints","Unlimited budget","Only past designs"], 0),
    "Strategy": ("Scope exceeds capacity; best move?", ["Prioritize must-haves","Do all items","Extend time w/o approval","Random order"], 0),
    "Social": ("To fix a misunderstanding, first:", ["Ask open questions","Repeat louder","Assume motives","Change topic"], 0),
}
SJT_TEMPLATES = {
    "Analytical": ("Stakeholder wants new metric mid-analysis; best first step?", ["Explain risks and finish current metric then plan follow-up","Switch now","Ignore request","Restart from scratch"], 0),
    "Mathematical": ("Teammate avoids algebra; best first step?", ["Pair and practice targeted problems with short feedback loops","Assign more tasks randomly","Ignore until review","Remove algebra from scope"], 0),
    "Verbal": ("Draft is long and unclear; best first step?", ["Cut repetition and surface main point in first sentence","Add qualifiers","Switch to passive voice","Ask for more length"], 0),
    "Memory": ("Scores drop after a week; best next step?", ["Add spaced retrieval sessions and track recall rate","Increase rereading","Stop practice","Switch topics"], 0),
    "Spatial": ("Plan a machine layout; first step?", ["Sketch views to scale and mark constraints","Move items immediately","Rely on memory","Skip visuals"], 0),
    "Creativity": ("Team is fixated on first idea; next step?", ["Time-box alternatives and defer judgment","Debate forever","Vote immediately","Defer to hierarchy"], 0),
    "Strategy": ("Dependency slips 1 week; best response?", ["Replan critical path and inform stakeholders","Do nothing","Blame the team","Add overtime without plan"], 0),
    "Social": ("Two teammates disagree; best move?", ["Facilitate a short session to clarify goals/options/criteria","Pick a side","Escalate first","Email later"], 0),
}

def load_current_items() -> List[Dict[str, Any]]:
    try:
        return qb.load_bank()
    except Exception:
        return []

def bank_json_path() -> Path:
    # Write into the installed package directory
    return Path(sc.__file__).parent / "bank.json"

def save_items_to_bank_json(items: List[Dict[str, Any]]) -> None:
    path = bank_json_path()
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} items to {path}")

def next_id(existing: set[str], prefix: str) -> str:
    n = 1
    while True:
        cand = f"{prefix}_{n}"
        if cand not in existing:
            return cand
        n += 1

def ensure_counts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_dom: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_dom[it["domain"]].append(it)

    out = list(items)
    existing = set(it["id"] for it in items)

    for dom in DOMAINS:
        dom_items = by_dom.get(dom, [])
        mcq = [it for it in dom_items if it["type"] == "MCQ"]
        sjt = [it for it in dom_items if it["type"] == "SJT"]
        opn = [it for it in dom_items if it["type"] == "OPEN"]

        need_mcq = max(0, TARGETS["MCQ_min"] - len(mcq))
        mcq_plus2 = sum(1 for it in mcq if int(it.get("difficulty", 0)) >= 2)
        need_plus2 = max(0, TARGETS["MCQ_plus2_min"] - mcq_plus2)

        diff_cycle = itertools.cycle([0, 1, 2, -1, 2, 1, 0])
        stem, opts, correct = MCQ_TEMPLATES[dom]
        for _ in range(need_mcq):
            d = 2 if need_plus2 > 0 else next(diff_cycle)
            if need_plus2 > 0: need_plus2 -= 1
            qid = next_id(existing, f"auto_{dom.lower()}_mcq")
            existing.add(qid)
            out.append({
                "id": qid, "domain": dom, "type": "MCQ",
                "stem": stem, "options": opts, "correct": correct,
                "difficulty": int(d), "discrimination": 1.2,
                "variant_group": f"{dom}:MCQ:{stem[:16]}", "mirror_of": None
            })

        need_sjt = max(0, TARGETS["SJT_min"] - len(sjt))
        sjt_diffs = itertools.cycle([-1, 0, 1, 0, -1, 1])
        s_stem, s_opts, s_correct = SJT_TEMPLATES[dom]
        for _ in range(need_sjt):
            d = next(sjt_diffs)
            qid = next_id(existing, f"auto_{dom.lower()}_sjt")
            existing.add(qid)
            out.append({
                "id": qid, "domain": dom, "type": "SJT",
                "stem": s_stem, "options": s_opts, "correct": s_correct,
                "difficulty": int(d), "discrimination": 1.0,
                "variant_group": f"{dom}:SJT:{s_stem[:16]}", "mirror_of": None
            })

        need_open = max(0, TARGETS["OPEN_min"] - len(opn))
        for _ in range(need_open):
            qid = next_id(existing, f"auto_{dom.lower()}_open")
            existing.add(qid)
            out.append({
                "id": qid, "domain": dom, "type": "OPEN",
                "stem": "Provide a concise plan with metric, timeline, validation, and a decision rule.",
                "options": [], "correct": None,
                "difficulty": 1, "discrimination": 1.0,
                "variant_group": f"{dom}:OPEN:auto", "mirror_of": None
            })

    return out

def main():
    items = load_current_items()
    # if load_bank returned nothing, start from an empty list
    new_items = ensure_counts(items)
    save_items_to_bank_json(new_items)

if __name__ == "__main__":
    main()
