#!/usr/bin/env python3
"""Assign provisional ``variant_group`` identifiers across bank shards."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator


DEFAULT_PATTERN = "data/**/*.{json,jsonl}"


def _expand_pattern(pattern: str) -> Iterator[str]:
    """Expand a glob pattern that may contain a single brace group."""

    brace_start = pattern.find("{")
    brace_end = pattern.find("}", brace_start + 1)
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        prefix = pattern[:brace_start]
        suffix = pattern[brace_end + 1 :]
        for option in pattern[brace_start + 1 : brace_end].split(","):
            yield from _expand_pattern(f"{prefix}{option}{suffix}")
        return
    yield pattern


def _iter_files(root: Path, pattern: str) -> list[Path]:
    files: set[Path] = set()
    for expanded in _expand_pattern(pattern):
        expanded_path = Path(expanded)
        if expanded_path.is_absolute():
            matches = glob.glob(str(expanded_path), recursive=True)
        else:
            matches = glob.glob(str(root / expanded), recursive=True)
        files.update(Path(m) for m in matches if Path(m).is_file())
    return sorted(files)


def _canonical_text(item: dict) -> str:
    for key in ("stem", "prompt", "text", "question"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    fallback = item.get("id")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return ""


def _compute_variant_group(item: dict) -> str:
    domain = str(item.get("domain") or "Unknown").strip() or "Unknown"
    if "type" in item and item["type"]:
        typ = str(item["type"]).strip() or "MCQ"
    else:
        typ = "OPEN" if "rubric" in item else "MCQ"
    diff = item.get("difficulty")
    if diff is None:
        diff = item.get("level", 0)
    text = _canonical_text(item)
    key = item.get("id") or text
    token = hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:10]
    return f"{domain}:{typ}:{diff}:{token}"


def _looks_like_item(node: object) -> bool:
    if not isinstance(node, dict):
        return False
    if "id" not in node:
        return False
    keys = {"prompt", "stem", "text", "question", "type", "options", "choices", "rubric"}
    return any(k in node for k in keys)


def _collect_items(node: object) -> list[dict]:
    items: list[dict] = []

    def walk(obj: object) -> None:
        if isinstance(obj, list):
            for entry in obj:
                walk(entry)
            return
        if isinstance(obj, dict):
            if _looks_like_item(obj):
                items.append(obj)
            for value in obj.values():
                walk(value)

    walk(node)
    return items


def _load_items(path: Path) -> tuple[list[dict], Callable[[list[dict]], str], object]:
    if path.suffix.lower() == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        items = [json.loads(line) for line in lines if line.strip()]

        def writer(updated: list[dict]) -> str:
            return "\n".join(json.dumps(entry, ensure_ascii=False) for entry in updated) + "\n"

        return items, writer, items

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        def writer(updated: list[dict]) -> str:
            return json.dumps(updated, indent=2, ensure_ascii=False) + "\n"

        return raw, writer, raw

    if isinstance(raw, dict):
        items = _collect_items(raw)

        def writer(updated: list[dict]) -> str:
            # ``updated`` already mutates ``raw`` in place; serialize entire structure.
            return json.dumps(raw, indent=2, ensure_ascii=False) + "\n"

        return items, writer, raw

    raise ValueError(f"Unsupported bank schema in {path}")


def _apply_variant_groups(items: Iterable[dict]) -> tuple[int, int]:
    fixed = 0
    unchanged = 0
    for item in items:
        vg = item.get("variant_group")
        if vg is None or not str(vg).strip():
            item["variant_group"] = _compute_variant_group(item)
            fixed += 1
        else:
            unchanged += 1
    return fixed, unchanged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign provisional variant_group IDs across bank shards")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="glob pattern for bank files (default: %(default)s)")
    parser.add_argument("--apply", action="store_true", help="write patched files alongside their sources")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    files = _iter_files(root, args.pattern)
    if not files:
        print("No bank files matched the pattern.")
        return 0

    total_items = 0
    total_fixed = 0
    total_unchanged = 0

    for file_path in files:
        items, writer, _raw = _load_items(file_path)
        fixed, unchanged = _apply_variant_groups(items)
        total_items += len(items)
        total_fixed += fixed
        total_unchanged += unchanged

        output_path = Path(str(file_path) + ".patched" + file_path.suffix)
        if args.apply:
            output_path.write_text(writer(items), encoding="utf-8")
            action = f"wrote {output_path}"
        else:
            action = "dry run"
        print(f"{file_path}: total={len(items)} fixed={fixed} unchanged={unchanged} ({action})")

    print(
        f"Totals: files={len(files)} items={total_items} fixed={total_fixed} unchanged={total_unchanged}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
