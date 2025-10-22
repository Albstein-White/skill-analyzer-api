from __future__ import annotations

from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

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
        "GOD_PROBE_MIN_FIRST",
        "GOD_PROBE_MIN_SECOND",
        "OPEN_CAL_A",
        "OPEN_CAL_B",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("OPEN_CAL_A", "1.0")
    monkeypatch.setenv("OPEN_CAL_B", "0.0")
    _, app_module = _reload_app(tmp_path)
    return TestClient(app_module.app)


def _run(client: TestClient, params: Dict[str, Any]) -> Dict[str, Any]:
    response = client.get("/dev/run", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_god_probe_mixed_prod(dev_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import api.dev as dev_mod
    import skill_core.llm_bridge as llm_bridge
    import skill_core.heuristics as heuristics
    import skill_core.scoring as scoring
    import skill_core.engine as engine_mod

    def biased_heuristic(_: str) -> float:
        domain = getattr(dev_mod, "_current_open_domain", None)
        return 0.90 if domain == "Analytical" else 0.75

    monkeypatch.setattr(dev_mod, "heuristic_open_score", biased_heuristic, raising=False)
    monkeypatch.setattr(llm_bridge, "heuristic_open_score", biased_heuristic, raising=True)
    monkeypatch.setattr(heuristics, "heuristic_open_score", biased_heuristic, raising=True)
    monkeypatch.setattr(dev_mod, "_current_open_domain", None, raising=False)

    original_score_item = scoring.score_item

    def patched_score_item(item, answer):
        item_type = str(getattr(item, "type", "")).upper()
        if item_type == "OPEN":
            domain = getattr(item, "domain", None)
            setattr(dev_mod, "_current_open_domain", domain)
            credit = 0.90 if domain == "Analytical" else 0.75
            return credit, {"type": "OPEN", "score": credit, "used_prompt_stub": False}
        setattr(dev_mod, "_current_open_domain", None)
        return original_score_item(item, answer)

    monkeypatch.setattr(scoring, "score_item", patched_score_item)
    monkeypatch.setattr(engine_mod, "score_item", patched_score_item)

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
        "PROD_GOD_RUBRIC0": "0.85",
        "PROD_GOD_RUBRIC1": "0.85",
        "GOD_PROBE_MIN_FIRST": "0.85",
        "GOD_PROBE_MIN_SECOND": "0.85",
    }

    payload = _run(dev_client, params)

    assert payload["steps"] == 160
    assert payload["effective_steps"] == 160
    assert payload["open_used"] >= 8
    assert payload["tiers"] == ["SS"] * 8

    probe_block = payload.get("probe")
    if not probe_block:
        god_meta = (payload.get("god_probe") or {}).get("Analytical", {})
        probe_block = {
            "enabled": bool(payload.get("god_probe_enabled")),
            "domain": "Analytical",
            "first_rubric": god_meta.get("first_rubric"),
            "first_difficulty": god_meta.get("first_difficulty"),
            "second_rubric": god_meta.get("probe2_rubric"),
            "probe2_served": god_meta.get("probe2_served"),
            "probe_reason": god_meta.get("probe_reason"),
        }
    assert probe_block["enabled"] is True
    assert probe_block["domain"] == "Analytical"
    assert probe_block["probe2_served"] is False
    assert probe_block["probe_reason"] == "probe_blocked_below_ss"
