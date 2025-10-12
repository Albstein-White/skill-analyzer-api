from __future__ import annotations

import json
import argparse
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


def audit_items(
    items: Iterable[Item],
    *,
    min_obj: int | None = None,
    min_open: int | None = None,
    require_variant_group: bool | None = None,
) -> dict[str, object]:
    coverage: dict[str, dict[str, object]] = {domain: _blank_domain() for domain in DOMAINS}
    totals = {"MCQ": 0, "SJT": 0, "OPEN": 0, "missing_variant_group": 0}

    min_obj = config.BANK_MIN_PER_BUCKET_OBJ if min_obj is None else int(min_obj)
    min_open = config.BANK_MIN_PER_BUCKET_OPEN if min_open is None else int(min_open)
    require_variant_group = (
        config.BANK_EXPECT_VARIANT_GROUP
        if require_variant_group is None
        else bool(require_variant_group)
    )

    for item in items:
        domain_data = coverage.setdefault(item.domain, _blank_domain())

        if not getattr(item, "variant_group", None):
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
            if mcq.get(lvl, 0) < min_obj:
                warnings.append(
                    f"{domain} MCQ level {lvl:+d} has {mcq.get(lvl, 0)} (<{min_obj})"
                )
            if sjt.get(lvl, 0) < min_obj:
                warnings.append(
                    f"{domain} SJT level {lvl:+d} has {sjt.get(lvl, 0)} (<{min_obj})"
                )

        for lvl in OPEN_BUCKETS:
            if open_map.get(lvl, 0) < min_open:
                warnings.append(
                    f"{domain} OPEN level {lvl:+d} has {open_map.get(lvl, 0)} (<{min_open})"
                )

        missing_vg = data["missing_variant_group"]  # type: ignore[assignment]
        if missing_vg and require_variant_group:
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
    parser = argparse.ArgumentParser(description="Audit bank coverage for adaptive ladder readiness")
    parser.add_argument("--allow-warn", dest="allow_warn", action="store_true", help="exit with status 0 even when warnings are present")
    parser.add_argument("--min-obj", type=int, default=None, help="override minimum MCQ/SJT items per difficulty bucket")
    parser.add_argument("--min-open", type=int, default=None, help="override minimum OPEN items per difficulty bucket")
    parser.add_argument("--no-vg-required", action="store_true", help="suppress missing variant_group warnings")
    args = parser.parse_args(_argv)

    items = load_bank()
    summary = audit_items(
        items,
        min_obj=args.min_obj,
        min_open=args.min_open,
        require_variant_group=False if args.no_vg_required else None,
    )
    print_report(summary)
    write_summary(summary)
    allow_warn = args.allow_warn or config.BANK_AUDIT_ALLOW_WARN
    exit_code = 0 if (allow_warn or not summary["warnings"]) else 2
    print(
        f"AUDIT: exit={exit_code} warnings={len(summary['warnings'])} allow_warn={bool(allow_warn)}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
