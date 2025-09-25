
from __future__ import annotations
from typing import Dict
SYNERGY: Dict[tuple[str,str], float] = {}
def _pair(a,b): return tuple(sorted((a,b)))
def _init():
    pairs = {
        ("Analytical","Strategy"): 0.8,
        ("Analytical","Mathematical"): 0.7,
        ("Verbal","Social"): 0.7,
        ("Creativity","Strategy"): 0.6,
        ("Spatial","Mathematical"): 0.6,
    }
    for k,v in pairs.items(): SYNERGY[_pair(*k)] = v
_init()
def synergy_sum(domain_scores: dict[str,float]) -> float:
    keys = list(domain_scores.keys()); total = 0.0
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            a,b = keys[i], keys[j]
            w = SYNERGY.get(_pair(a,b),0.0)
            if w<=0: continue
            if domain_scores[a]>=68 and domain_scores[b]>=68:
                total += w * (min(domain_scores[a]-68,32)/32 + min(domain_scores[b]-68,32)/32) * 4.0
    return min(total, 20.0)
