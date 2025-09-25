
from __future__ import annotations
import json, importlib.resources as ir
from typing import List
from .types import Item
DOMAINS = ["Analytical","Mathematical","Verbal","Memory","Spatial","Creativity","Strategy","Social"]
def load_bank() -> List[Item]:
    data = ir.files(__package__).joinpath("data/bank.json").read_text(encoding="utf-8")
    raw = json.loads(data)
    return [Item(**r) for r in raw]
