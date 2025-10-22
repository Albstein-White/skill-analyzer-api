from collections import defaultdict
from typing import Any, Dict

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
    ):
        monkeypatch.delenv(env_var, raising=False)
    _, app_module = _reload_app(tmp_path)
    return TestClient(app_module.app)


def _run(client: TestClient, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.get("/dev/run", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _patch_tier(monkeypatch: pytest.MonkeyPatch, label: str) -> None:
    import skill_core.engine as engine_mod
    import api.app as app_mod

    original = engine_mod.AdaptiveSession._current_tier_label

    def patched(self, domain: str) -> str:
        if domain == "Analytical":
            return label
        return original(self, domain)

    monkeypatch.setattr(engine_mod.AdaptiveSession, "_current_tier_label", patched)
    if hasattr(app_mod, "AdaptiveSession"):
        monkeypatch.setattr(app_mod.AdaptiveSession, "_current_tier_label", patched)


def _patch_open_scores(
    monkeypatch: pytest.MonkeyPatch, first: float, subsequent: float
) -> None:
    import skill_core.engine as engine_mod
    import skill_core.scoring as scoring_mod
    import api.dev as dev_mod

    original = scoring_mod.score_item
    counts: Dict[str, int] = defaultdict(int)

    def patched(item, answer):
        item_type = str(getattr(item, "type", "")).upper()
        if item_type == "OPEN":
            domain = getattr(item, "domain", "")
            counts[domain] += 1
            credit = first if counts[domain] == 1 else subsequent
            return credit, {"type": "OPEN", "score": credit}
        return original(item, answer)

    monkeypatch.setattr(scoring_mod, "score_item", patched)
    monkeypatch.setattr(engine_mod, "score_item", patched)
    monkeypatch.setattr(dev_mod, "score_item", patched)


def test_open_served_at_A_no_probe(dev_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_tier(monkeypatch, "A")

    payload = _run(dev_client, {"run": "long", "mode": "pass", "seed": 11})

    meta = (payload.get("god_probe") or {}).get("Analytical", {})
    open_tier = (payload.get("open_first_tier") or {}).get("Analytical")
    assert open_tier in {"A", "S"}
    assert meta.get("first_open_was_below_ss") is True
    assert meta.get("probe2_served") is False
    assert meta.get("first_rubric") is not None


def test_open_served_at_S_no_probe(dev_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_tier(monkeypatch, "S")

    payload = _run(dev_client, {"run": "long", "mode": "pass", "seed": 12})

    meta = (payload.get("god_probe") or {}).get("Analytical", {})
    open_tier = (payload.get("open_first_tier") or {}).get("Analytical")
    assert open_tier in {"A", "S"}
    assert meta.get("first_open_was_below_ss") is True
    assert meta.get("probe2_served") is False


def test_probe_requires_ss_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    import skill_core.engine as engine_mod

    monkeypatch.setenv("PROD_GOD_ENABLE", "1")
    monkeypatch.setenv("STAGING_PROFILE", "1")

    session = engine_mod.AdaptiveSession("long")
    domain = session.state.domains["Analytical"]
    domain.theta = 3.0
    domain.se = 0.1
    domain.b_stable = engine_mod.GOD_REQ_LEVEL
    domain.level_stats = {2: {"seen": 50, "correct": 45}}
    domain.open_count = 1
    domain.obj_count = max(domain.obj_count, 6)

    session.first_open_rubric["Analytical"] = engine_mod.GOD_PROBE_MIN_FIRST + 0.05
    session.first_open_difficulty["Analytical"] = engine_mod.GOD_PROBE_DIFFICULTY
    session.open_first_tier["Analytical"] = "SS"
    session.first_open_was_below_ss["Analytical"] = False
    session.god_probe_used["Analytical"] = False
    session.god_probe_pending["Analytical"] = False
    session.god_probe_enabled = True

    monkeypatch.setattr(
        engine_mod,
        "god_thresholds",
        lambda: {
            "min_norm": 0.0,
            "max_se": 1.0,
            "min_l2_seen": 0,
            "min_l2_acc": 0.0,
            "min_open": 0,
            "rubric0": engine_mod.GOD_PROBE_MIN_FIRST,
            "rubric1": engine_mod.GOD_PROBE_MIN_SECOND,
        },
    )

    st = session._policy_state()
    assert session.policy._should_queue_god_probe(st, "Analytical") is True

    session.first_open_was_below_ss["Analytical"] = True
    st = session._policy_state()
    assert session.policy._should_queue_god_probe(st, "Analytical") is False


def test_high_rubric_at_A_still_no_probe(
    dev_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tier(monkeypatch, "A")
    _patch_open_scores(monkeypatch, first=0.9, subsequent=0.9)

    payload = _run(
        dev_client,
        {
            "run": "long",
            "mode": "pass",
            "seed": 11,
            "PROD_GOD_ENABLE": "1",
        },
    )

    meta = (payload.get("god_probe") or {}).get("Analytical", {})
    open_tier = (payload.get("open_first_tier") or {}).get("Analytical")
    assert open_tier in {"A", "S"}
    assert meta.get("first_open_was_below_ss") is True
    assert meta.get("probe2_served") is False
    assert "probe_blocked_below_ss" in (meta.get("reasons") or [])
