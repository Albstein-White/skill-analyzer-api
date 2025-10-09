from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from . import config
from .question_bank import DOMAINS, load_bank
from .types import Item

OBJ_BUCKETS: tuple[int, ...] = (-2, -1, 0, 1, 2)
OPEN_BUCKETS: tuple[int, ...] = (-1, 0, 1)


def _blank_domain() -> dict[str, object]:
    return {
        "MCQ": {lvl: 0 for lvl in OBJ_BUCKETS},
        "SJT": {lvl: 0 for lvl in OBJ_BUCKETS},
        "OPEN": {lvl: 0 for lvl in OPEN_BUCKETS},
        "missing_variant_group": 0,
    }


def audit_items(items: Iterable[Item]) -> dict[str, object]:
    coverage: dict[str, dict[str, object]] = {domain: _blank_domain() for domain in DOMAINS}
    totals = {"MCQ": 0, "SJT": 0, "OPEN": 0, "missing_variant_group": 0}

    for item in items:
        domain_data = coverage.setdefault(item.domain, _blank_domain())

        if not item.variant_group and config.BANK_EXPECT_VARIANT_GROUP:
            domain_data["missing_variant_group"] += 1
            totals["missing_variant_group"] += 1

        bucket_map: dict[int, int] | None = None
        if item.type in ("MCQ", "SJT"):
            bucket_map = domain_data[item.type]  # type: ignore[index]
            totals[item.type] += 1
        elif item.type == "OPEN":
            bucket_map = domain_data[item.type]  # type: ignore[index]
            totals["OPEN"] += 1

        if bucket_map is not None and isinstance(item.difficulty, int):
            if item.difficulty not in bucket_map:
                bucket_map[item.difficulty] = 0
            bucket_map[item.difficulty] += 1

    warnings: list[str] = []
    for domain, data in coverage.items():
        mcq = data["MCQ"]  # type: ignore[assignment]
        sjt = data["SJT"]  # type: ignore[assignment]
        open_map = data["OPEN"]  # type: ignore[assignment]

        for lvl in OBJ_BUCKETS:
            if mcq.get(lvl, 0) < config.BANK_MIN_PER_BUCKET_OBJ:
                warnings.append(
                    f"{domain} MCQ level {lvl:+d} has {mcq.get(lvl, 0)} (<{config.BANK_MIN_PER_BUCKET_OBJ})"
                )
            if sjt.get(lvl, 0) < config.BANK_MIN_PER_BUCKET_OBJ:
                warnings.append(
                    f"{domain} SJT level {lvl:+d} has {sjt.get(lvl, 0)} (<{config.BANK_MIN_PER_BUCKET_OBJ})"
                )

        for lvl in OPEN_BUCKETS:
            if open_map.get(lvl, 0) < config.BANK_MIN_PER_BUCKET_OPEN:
                warnings.append(
                    f"{domain} OPEN level {lvl:+d} has {open_map.get(lvl, 0)} (<{config.BANK_MIN_PER_BUCKET_OPEN})"
                )

        missing_vg = data["missing_variant_group"]  # type: ignore[assignment]
        if missing_vg and config.BANK_EXPECT_VARIANT_GROUP:
            warnings.append(f"{domain} has {missing_vg} items missing variant_group")

    summary = {"coverage": coverage, "warnings": warnings, "totals": totals}
    return summary


def _format_row(label: str, buckets: Iterable[int], data: dict[int, int]) -> str:
    parts = [label]
    for lvl in buckets:
        parts.append(f"{lvl:+d}:{data.get(lvl, 0):3d}")
    return "  ".join(parts)


def print_report(summary: dict[str, object]) -> None:
    coverage: dict[str, dict[str, object]] = summary["coverage"]  # type: ignore[assignment]
    print("=== Bank Coverage ===")
    for domain in sorted(coverage):
        data = coverage[domain]
        print(f"\nDomain: {domain}")
        print("  " + _format_row("MCQ ", OBJ_BUCKETS, data["MCQ"]))  # type: ignore[arg-type]
        print("  " + _format_row("SJT ", OBJ_BUCKETS, data["SJT"]))  # type: ignore[arg-type]
        print("  " + _format_row("OPEN", OPEN_BUCKETS, data["OPEN"]))  # type: ignore[arg-type]
        missing_vg = data["missing_variant_group"]  # type: ignore[index]
        if missing_vg:
            print(f"    missing_variant_group: {missing_vg}")

    warnings: list[str] = summary["warnings"]  # type: ignore[assignment]
    if warnings:
        print("\nWarnings:")
        for msg in warnings:
            print(f" - {msg}")
    else:
        print("\nNo warnings.")

    totals = summary["totals"]
    print("\nTotals:", totals)


def write_summary(summary: dict[str, object], path: Path = Path("/tmp/bank_audit.json")) -> str:
    text = json.dumps(summary, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return text


def main(_argv: list[str] | None = None) -> int:
    items = load_bank()
    summary = audit_items(items)
    print_report(summary)
    write_summary(summary)
    return 2 if summary["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
