
from __future__ import annotations
from typing import Dict, List
from .types import Item, Answer
def count_traps(items: List[Item], answers: Dict[str, Answer]) -> int:
    cnt = 0
    for it in items:
        if not it.is_trap: continue
        ans = answers.get(it.id)
        if not ans: continue
        if it.trap_flag_index is not None:
            try:
                if int(ans.value) == it.trap_flag_index: cnt += 1; continue
            except: pass
        if it.type == "SR":
            try:
                if int(ans.value) == 5: cnt += 1
            except: pass
    return cnt
def consistency_index(items: List[Item], answers: Dict[str, Answer]) -> float:
    pairs = [(it, next((x for x in items if x.mirror_of==it.id), None)) 
             for it in items if it.type=="SR" and it.mirror_of is None]
    scores = []
    for a,b in pairs:
        if not b: continue
        va = _likert(answers.get(a.id)); vb = _likert(answers.get(b.id))
        if va is None or vb is None: continue
        diff = abs(va - (6 - vb))
        scores.append(max(0.0, 1.0 - diff/4.0))
    if not scores: return 1.0
    return sum(scores)/len(scores)
def _likert(ans: Answer|None):
    if not ans: return None
    try: v = int(ans.value)
    except: return None
    return v if 1<=v<=5 else None
