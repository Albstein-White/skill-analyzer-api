from __future__ import annotations

from pathlib import Path

import skill_core.audit_bank as audit_bank
from skill_core import config
from skill_core.types import Item
from tests.conftest import build_synthetic_bank


def test_audit_flags_sparse_buckets(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BANK_MIN_PER_BUCKET_OBJ", 2, raising=False)
    monkeypatch.setattr(config, "BANK_MIN_PER_BUCKET_OPEN", 2, raising=False)

    bank = build_synthetic_bank(domains=["Analytical"], mcq_per_level=1, include_sr=False)

    summary = audit_bank.audit_items(bank)
    assert summary["warnings"], "expected sparse coverage warnings"
    joined = "\n".join(summary["warnings"])
    assert "Analytical MCQ level -2" in joined
    assert "Analytical OPEN level -1" in joined

    outfile = tmp_path / "bank_audit.json"
    text = audit_bank.write_summary(summary, path=outfile)
    assert outfile.read_text(encoding="utf-8").strip() == text


def test_audit_flags_missing_variant_group(monkeypatch):
    monkeypatch.setattr(config, "BANK_MIN_PER_BUCKET_OBJ", 0, raising=False)
    monkeypatch.setattr(config, "BANK_MIN_PER_BUCKET_OPEN", 0, raising=False)
    monkeypatch.setattr(config, "BANK_EXPECT_VARIANT_GROUP", True, raising=False)

    item = Item(id="a1", domain="Analytical", type="MCQ", text="t", difficulty=0)
    summary = audit_bank.audit_items([item])

    assert summary["coverage"]["Analytical"]["missing_variant_group"] == 1
    assert any("missing variant_group" in w for w in summary["warnings"])


def test_main_returns_warning_exit(monkeypatch, capsys):
    monkeypatch.setattr(config, "BANK_MIN_PER_BUCKET_OBJ", 2, raising=False)
    monkeypatch.setattr(config, "BANK_MIN_PER_BUCKET_OPEN", 2, raising=False)

    bank = build_synthetic_bank(domains=["Analytical"], mcq_per_level=1, include_sr=False)
    monkeypatch.setattr(audit_bank, "load_bank", lambda: bank)

    exit_code = audit_bank.main([])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Analytical" in captured.out
    assert Path("/tmp/bank_audit.json").exists()
