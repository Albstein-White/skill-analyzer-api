#!/usr/bin/env python3
"""Assign provisional variant_group identifiers to bank items."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Tuple

DEFAULT_BANK_PATH = Path(__file__).resolve().parents[1] / "skill_core" / "data" / "bank.json"


def _canonical_text(item: dict) -> str:
    for key in ("stem", "prompt", "text"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return str(item.get("id", ""))


def _compute_variant_group(item: dict) -> str:
    base_text = _canonical_text(item)
    token = hashlib.sha1(base_text.encode("utf-8")).hexdigest()[:10]
    return (
        f"{item.get('domain','')}:{item.get('type','')}:"
        f"{item.get('difficulty',0)}:{token}"
    )


def _load_bank(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("bank file must contain a list")
    return [dict(entry) for entry in data]


def _apply_fixes(entries: Iterable[dict]) -> Tuple[list[dict], int, int]:
    updated: list[dict] = []
    fixed = 0
    unchanged = 0
    for entry in entries:
        record = dict(entry)
        vg = record.get("variant_group")
        if vg is None or not str(vg).strip():
            record["variant_group"] = _compute_variant_group(record)
            fixed += 1
        else:
            unchanged += 1
        updated.append(record)
    return updated, fixed, unchanged


def main(_argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign provisional variant_group IDs to bank items")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK_PATH, help="path to bank JSON file")
    parser.add_argument("--apply", action="store_true", help="write patched bank alongside the source file")
    args = parser.parse_args(_argv)

    bank_path = args.bank
    if not bank_path.exists():
        raise SystemExit(f"bank file not found: {bank_path}")

    entries = _load_bank(bank_path)
    updated, fixed, unchanged = _apply_fixes(entries)
    total = len(updated)

    output_path = Path(f"{bank_path}.patched.json")
    if args.apply:
        output_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote patched bank to {output_path}")
    else:
        print(f"Dry run. Use --apply to write {output_path.name}")

    print(f"Total items: {total}  fixed: {fixed}  unchanged: {unchanged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
