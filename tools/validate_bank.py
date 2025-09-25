from __future__ import annotations
from collections import defaultdict, Counter
import os
from skill_core.question_bank import load_bank, DOMAINS

# Configurable targets; defaults match your current bank
TARGETS = {
    "MCQ_min": int(os.getenv("TARGET_MCQ_MIN", 10)),
    "SJT_min": int(os.getenv("TARGET_SJT_MIN", 2)),
    "OPEN_min": int(os.getenv("TARGET_OPEN_MIN", 2)),
    "MCQ_plus2_min": int(os.getenv("TARGET_MCQ_PLUS2_MIN", 2)),
}

def main():
    items = load_bank()
    by_dom = defaultdict(list)
    for it in items:
        by_dom[it.domain].append(it)

    print(f"Targets per domain: ≥{TARGETS['MCQ_min']} MCQ (≥{TARGETS['MCQ_plus2_min']} at +2), "
          f"≥{TARGETS['SJT_min']} SJT, ≥{TARGETS['OPEN_min']} OPEN.\n")

    for d in DOMAINS:
        dom = by_dom[d]
        mcq = [it for it in dom if it.type == "MCQ"]
        sjt = [it for it in dom if it.type == "SJT"]
        opn = [it for it in dom if it.type == "OPEN"]
        plus2 = sum(1 for it in mcq if int(it.difficulty or 0) >= 2)

        # per-difficulty summary
        print(f"{d}: MCQ={len(mcq)} SJT={len(sjt)} OPEN={len(opn)}  diff(+2 MCQ)={plus2}")
        for diff in [-2, -1, 0, 1, 2]:
            m = sum(1 for it in mcq if int(it.difficulty or 0) == diff)
            s = sum(1 for it in sjt if int(it.difficulty or 0) == diff)
            print(f"  diff {diff:+}: MCQ {m:2d} | SJT {s:2d}")

        # deficits
        need_mcq = max(0, TARGETS["MCQ_min"] - len(mcq))
        need_sjt = max(0, TARGETS["SJT_min"] - len(sjt))
        need_open = max(0, TARGETS["OPEN_min"] - len(opn))
        need_p2  = max(0, TARGETS["MCQ_plus2_min"] - plus2)

        if need_mcq or need_sjt or need_open or need_p2:
            print(f"  → Add: MCQ {need_mcq}, SJT {need_sjt}, OPEN {need_open}, extra MCQ@+2 {need_p2}\n")
        else:
            print("  ✓ Meets targets\n")

if __name__ == "__main__":
    main()
