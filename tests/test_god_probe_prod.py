from __future__ import annotations

from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

from tests.test_api_flows import _reload_app
from skill_core.question_bank import DOMAINS


@pytest.fixture
def dev_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("STAGING_PROFILE", "1")
    # ensure production GOD feature starts disabled with default knobs
    for env_var in (
        "PROD_GOD_ENABLE",
        "PROD_GOD_MIN_NORM",
        "PROD_GOD_MAX_SE",
        "PROD_GOD_MIN_L2_SEEN",
        "PROD_GOD_MIN_L2_ACC",
        "PROD_GOD_MIN_OPEN",
        "PROD_GOD_RUBRIC0",
        "PROD_GOD_RUBRIC1",
        "GOD_PROBE_EXEMPT_FROM_CAP",
    ):
        monkeypatch.delenv(env_var, raising=False)
    _, app_module = _reload_app(tmp_path)
    return TestClient(app_module.app)


def _run(client: TestClient, params: Dict[str, Any]) -> Dict[str, Any]:
    response = client.get("/dev/run", params=params)
    assert response.status_code == 200
    return response.json()


def test_prod_default_control(dev_client: TestClient) -> None:
    payload = _run(dev_client, {"run": "long", "mode": "pass", "seed": 11})

    assert payload["steps"] == payload["effective_steps"]
    assert 100 <= payload["steps"] <= 160
    assert payload["open_used"] >= 8
    assert payload.get("god_probe_enabled") is False
    assert payload.get("extra_steps_exempt", 0) == 0
    assert str(payload.get("cap", "")).endswith("/160")
    tiers = payload.get("tiers") or []
    assert tiers and all(tier == "SS" for tier in tiers)

    for meta in (payload.get("god_probe") or {}).values():
        assert meta.get("probe2_served") is False
        assert meta.get("probe_reason") in (None, "blocked_progress")


def test_prod_god_awarded(dev_client: TestClient) -> None:
    params = {
        "run": "long",
        "mode": "pass",
        "seed": 11,
        "PROD_GOD_ENABLE": "1",
        "PROD_GOD_MIN_NORM": "9.4",
        "PROD_GOD_MAX_SE": "0.6",
        "PROD_GOD_MIN_L2_SEEN": "1",
        "PROD_GOD_MIN_L2_ACC": "0.7",
        "PROD_GOD_MIN_OPEN": "1",
        "PROD_GOD_RUBRIC0": "0.6",
        "PROD_GOD_RUBRIC1": "0.6",
    }
    payload = _run(dev_client, params)

    assert payload["steps"] == payload["effective_steps"]
    assert 100 <= payload["steps"] <= 160
    assert payload["open_used"] >= 8
    assert payload.get("extra_steps_exempt", 0) == 0
    assert payload.get("god_probe_enabled") is True
    tiers = payload.get("tiers") or []
    assert tiers, "expected tiers in payload"

    tier_map = dict(zip(DOMAINS, payload.get("tiers") or []))
    assert "GOD" in tier_map.values()
    for domain, meta in (payload.get("god_probe") or {}).items():
        tier = tier_map.get(domain)
        first_tier = meta.get("open_first_tier")
        if first_tier in {"SS", "GOD"}:
            assert meta.get("probe2_served") is True, domain
            assert meta.get("probe_reason") == "god_awarded", domain
            assert meta.get("passed") is True
            assert tier == "GOD"
        else:
            assert meta.get("probe2_served") is False, domain
            assert meta.get("probe_reason") in {None, "probe_blocked_below_ss"}, domain
            if meta.get("probe_reason") != "probe_blocked_below_ss":
                assert meta.get("first_open_was_below_ss") is True

    overrides = (
        (payload.get("debug") or {})
        .get("god_thresholds_echo", {})
        .get("prod_overrides_applied", {})
    )
    assert overrides.get("enabled") is True
    assert overrides.get("min_norm") == pytest.approx(9.4)
    assert overrides.get("max_se") == pytest.approx(0.6)
    assert overrides.get("min_l2_seen") == 1
    assert overrides.get("rubric0") == pytest.approx(0.6)
    assert overrides.get("rubric1") == pytest.approx(0.6)


def test_prod_god_denied_high_rubric(dev_client: TestClient) -> None:
    params = {
        "run": "long",
        "mode": "pass",
        "seed": 11,
        "PROD_GOD_ENABLE": "1",
        "PROD_GOD_MIN_NORM": "9.4",
        "PROD_GOD_MAX_SE": "0.6",
        "PROD_GOD_MIN_L2_SEEN": "1",
        "PROD_GOD_MIN_L2_ACC": "0.7",
        "PROD_GOD_MIN_OPEN": "1",
        "PROD_GOD_RUBRIC0": "1.01",
        "PROD_GOD_RUBRIC1": "1.01",
    }
    payload = _run(dev_client, params)

    assert payload["steps"] == payload["effective_steps"]
    assert 100 <= payload["steps"] <= 160
    assert payload["open_used"] >= 8
    assert payload.get("god_probe_enabled") is True
    tiers = payload.get("tiers") or []
    assert tiers and all(tier == "SS" for tier in tiers)

    for domain, meta in (payload.get("god_probe") or {}).items():
        first_tier = meta.get("open_first_tier")
        assert meta.get("probe2_served") is False, domain
        if first_tier in {"SS", "GOD"}:
            assert meta.get("probe_reason") == "blocked_progress"
        else:
            assert meta.get("probe_reason") in (None, "probe_blocked_below_ss")
            if meta.get("probe_reason") != "probe_blocked_below_ss":
                assert meta.get("first_open_was_below_ss") is True


def test_prod_long_fail_blocks_open(dev_client: TestClient) -> None:
    payload = _run(
        dev_client,
        {"run": "long", "mode": "fail", "seed": 11, "PROD_GOD_ENABLE": "1"},
    )

    assert payload["steps"] == payload["effective_steps"] == 144
    assert payload["open_used"] == 0
    assert payload.get("stop_reason") in (None, "")
    assert payload.get("god_probe_enabled") is True

    for domain, meta in (payload.get("god_probe") or {}).items():
        assert meta.get("probe2_served") is False, domain
        assert meta.get("probe_reason") == "blocked_progress"


def test_short_run_sanity(dev_client: TestClient) -> None:
    payload = _run(dev_client, {"run": "short", "mode": "pass", "seed": 11})

    assert payload["steps"] == payload["effective_steps"] == 40
    assert payload["open_used"] == 0
    tiers = payload.get("tiers") or []
    assert tiers and all(tier == "A" for tier in tiers)
