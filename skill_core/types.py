
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
ItemType = Literal["SR","MCQ","SJT","OPEN"]
@dataclass
class Item:
    id: str; domain: str; type: ItemType; text: str
    options: Optional[List[str]] = None
    correct: Optional[int] = None
    weight: float = 1.0
    is_trap: bool = False
    mirror_of: Optional[str] = None
    trap_flag_index: Optional[int] = None
    difficulty: int = 0
    discrimination: float = 1.0
    variant_group: Optional[str] = None
@dataclass
class Answer:
    item_id: str; value: str; rt_sec: Optional[float] = None
@dataclass
class DomainScore:
    domain: str
    theta: float
    se: float
    norm_score: float
    tier: str
    rarity: str
    reliability: Optional[Dict[str, object]] = None
@dataclass
class HiddenSkill:
    domain: str; confidence: Literal["Low","Medium","High"]; reason: str
@dataclass
class Result:
    run_type: Literal["short","long"]
    domain_scores: List[DomainScore]
    top_skills: List[str]
    hidden_skills: List[HiddenSkill]
    traps_tripped: int
    consistency: float
    synergy_boost: float
    unique_award: Optional[str] = None
    summary: Dict[str, float] = field(default_factory=dict)
    plan: List[Dict[str, object]] = field(default_factory=list)
    audit_events: List[Dict[str, object]] = field(default_factory=list)
