from __future__ import annotations

from typing import Dict, Any

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from skill_core.question_bank import DOMAINS
from tests.test_api_flows import _reload_app


@pytest.fixture
def dev_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("STAGING_PROFILE", "1")
    for env_var in (
        "PROD_GOD_ENABLE",
        "PROD_GOD_MIN_NORM",
        "PROD_GOD_MAX_SE",
        "PROD_GOD_MIN_L2_SEEN",
        "PROD_GOD_MIN_L2_ACC",
        "PROD_GOD_MIN_OPEN",
        "PROD_GOD_RUBRIC0",
        "PROD_GOD_RUBRIC1",
    ):
        monkeypatch.delenv(env_var, raising=False)
    _, app_module = _reload_app(tmp_path)
    return TestClient(app_module.app)


def _run(client: TestClient, params: Dict[str, Any]) -> Dict[str, Any]:
    response = client.get("/dev/run", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_god_probe_mixed_prod(dev_client: TestClient) -> None:
    params = {
        "run": "long",
        "mode": "pass",
        "seed": 12,
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

    tier_map = dict(zip(DOMAINS, payload.get("tiers") or []))
    assert "GOD" in tier_map.values()
    assert "SS" in tier_map.values()

    god_meta = payload.get("god_probe") or {}
    for domain, meta in god_meta.items():
        first_tier = meta.get("open_first_tier")
        if tier_map.get(domain) == "GOD":
            assert first_tier in {"SS", "GOD"}
            assert meta.get("probe2_served") is True
            assert meta.get("probe_reason") == "god_awarded"
            assert meta.get("passed") is True
        else:
            assert tier_map.get(domain) == "SS"
            assert meta.get("probe2_served") is False
            if first_tier in {"SS", "GOD"}:
                assert meta.get("probe_reason") == "blocked_progress"
            else:
                assert meta.get("first_open_was_below_ss") is True
