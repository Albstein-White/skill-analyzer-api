import os
import os
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

import skill_core.config as cfg
from tests.test_api_flows import _reload_app


def _dev_run(tmp_path, params: Dict[str, Any]) -> Dict[str, Any]:
    os.environ["STAGING_PROFILE"] = "1"
    try:
        _, app_module = _reload_app(tmp_path)
        with TestClient(app_module.app) as client:
            response = client.get("/dev/run", params=params)
        assert response.status_code == 200
        return response.json()
    finally:
        os.environ.pop("STAGING_PROFILE", None)
        _reload_app(tmp_path)


def test_long_early_stop_on_varies_counts(tmp_path):
    payload = _dev_run(
        tmp_path,
        {
            "run": "long",
            "mode": "pass",
            "seed": 11,
            "PROD_GOD_ENABLE": "1",
        },
    )

    steps = payload["steps"]
    assert payload["steps"] == payload["effective_steps"]
    assert 100 <= steps <= cfg.CAP_LONG
    per_domain = payload.get("per_domain") or {}
    assert per_domain, "per_domain telemetry missing"

    # ensure at least one domain stopped early and stayed under the objective cap
    assert any(meta.get("early_stop") for meta in per_domain.values())
    for meta in per_domain.values():
        obj_used = meta.get("obj_used", 0)
        if meta.get("early_stop"):
            assert obj_used >= cfg.LONG_OBJ_MIN
            assert obj_used < cfg.LONG_OBJ_MAX

    # OPEN gating remains active: at least one domain delivered an OPEN
    assert payload.get("open_used", 0) >= 1

    open_first = payload.get("open_first_tier") or {}
    below_ss = payload.get("first_open_was_below_ss") or {}
    god_probe_meta = payload.get("god_probe") or {}
    for domain, first_tier in open_first.items():
        if first_tier and first_tier not in {"SS", "GOD"}:
            assert below_ss.get(domain) is True
            probe_meta = god_probe_meta.get(domain, {})
            assert probe_meta.get("probe2_served") is False


def test_long_early_stop_off_reaches_caps(tmp_path):
    payload = _dev_run(
        tmp_path,
        {
            "run": "long",
            "mode": "pass",
            "seed": 11,
            "TEST_EARLY_STOP": "off",
        },
    )

    steps = payload["steps"]
    assert payload["effective_steps"] == steps
    per_domain = payload.get("per_domain") or {}
    assert per_domain
    assert steps >= cfg.LONG_OBJ_MAX * len(per_domain)
    for meta in per_domain.values():
        assert meta.get("obj_used") == cfg.LONG_OBJ_MAX
        assert not meta.get("early_stop")
