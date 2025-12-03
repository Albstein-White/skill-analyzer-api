import os

import os

import pytest

from fastapi.testclient import TestClient

from tests.test_api_flows import _reload_app


def _client(tmp_path):
    os.environ["STAGING_PROFILE"] = "1"
    storage, app_module = _reload_app(tmp_path)
    return TestClient(app_module.app)


def test_dev_run_god_overrides(tmp_path):
    client = _client(tmp_path)
    try:
        prod = client.get("/dev/run", params={"run": "long", "mode": "pass", "seed": 11})
        assert prod.status_code == 200
        prod_payload = prod.json()
        tiers = prod_payload.get("tiers", [])
        assert tiers, "expected tiers in prod payload"
        assert all(tier != "GOD" for tier in tiers)
        reasons = prod_payload.get("reasons", {})
        assert isinstance(reasons, dict)
        allowed = {
            "norm_low",
            "se_high",
            "open_count_low",
            "open_rubric_low",
            "open_blocked_progress",
            "open_blocked_fail_fast",
            "no_items",
            "probe_blocked_below_ss",
        }
        for dom, flags in reasons.items():
            assert set(flags or []) & allowed, dom

        staging = client.get(
            "/dev/run",
            params={
                "run": "long",
                "mode": "pass",
                "seed": 12,
                "TEST_MODE": 1,
                "TEST_GOD_MIN_NORM": 0.0,
                "TEST_GOD_MAX_SE": 1.0,
                "TEST_GOD_MIN_L2_SEEN": 0,
                "TEST_GOD_MIN_L2_ACC": 0.0,
                "TEST_GOD_MIN_OPEN": 0,
                "TEST_OPEN_FULL": 1,
                "TEST_GOD_RUBRIC0": 0.0,
                "TEST_GOD_RUBRIC1": 0.0,
            },
        )
        assert staging.status_code == 200
        staging_payload = staging.json()
        assert staging_payload.get("tiers"), "expected tiers in staging payload"
        assert staging_payload.get("open_used") == 16
        for meta in (staging_payload.get("god_probe") or {}).values():
            assert meta.get("probe2_served") is False
            assert meta.get("first_open_was_below_ss") is True

        staging_god = client.get(
            "/dev/run",
            params={
                "run": "long",
                "mode": "god",
                "seed": 11,
                "TEST_MODE": 1,
                "TEST_GOD_MIN_NORM": 9.4,
                "TEST_GOD_MAX_SE": 0.60,
                "TEST_GOD_MIN_L2_SEEN": 12,
                "TEST_GOD_MIN_L2_ACC": 0.80,
                "TEST_GOD_MIN_OPEN": 2,
                "TEST_GOD_MIN_OPEN_RUBRIC": 0.1,
                "TEST_GOD_MIN_OPEN_RUBRIC_SEC": 0.05,
                "TEST_OPEN_FULL": 1,
            },
        )
        assert staging_god.status_code == 200
        staging_god_payload = staging_god.json()
        assert staging_god_payload.get("tiers")
        assert staging_god_payload.get("open_used") == 16
        debug_meta = staging_god_payload.get("debug") or {}
        thresholds = debug_meta.get("god_thresholds") or {}
        assert pytest.approx(0.1, rel=0.0, abs=1e-6) == thresholds.get("rubric0")
        assert pytest.approx(0.05, rel=0.0, abs=1e-6) == thresholds.get("rubric1")

        staging_god_fail = client.get(
            "/dev/run",
            params={
                "run": "long",
                "mode": "fail",
                "seed": 11,
                "TEST_MODE": 1,
                "TEST_GOD_MIN_NORM": 9.4,
                "TEST_GOD_MAX_SE": 0.60,
                "TEST_GOD_MIN_L2_SEEN": 12,
                "TEST_GOD_MIN_L2_ACC": 0.80,
                "TEST_GOD_MIN_OPEN": 2,
                "TEST_GOD_MIN_OPEN_RUBRIC": 0.1,
                "TEST_GOD_MIN_OPEN_RUBRIC_SEC": 0.05,
            },
        )
        assert staging_god_fail.status_code == 200
        staging_god_fail_payload = staging_god_fail.json()
        assert staging_god_fail_payload.get("open_used") == 0
        assert staging_god_fail_payload.get("stop_reason") in (None, "")

        realistic = client.get(
            "/dev/run",
            params={
                "run": "long",
                "mode": "pass",
                "seed": 11,
                "TEST_MODE": 1,
                "TEST_GOD_MIN_NORM": 9.4,
                "TEST_GOD_MAX_SE": 0.60,
                "TEST_GOD_MIN_L2_SEEN": 12,
                "TEST_GOD_MIN_L2_ACC": 0.80,
                "TEST_GOD_MIN_OPEN": 16,
                "TEST_GOD_RUBRIC0": 0.70,
                "TEST_GOD_RUBRIC1": 0.65,
                "TEST_GOD_MIN_OPEN_RUBRIC": 0.70,
                "TEST_GOD_MIN_OPEN_RUBRIC_SEC": 0.65,
                "TEST_OPEN_FULL": 1,
            },
        )
        assert realistic.status_code == 200
        realistic_payload = realistic.json()
        assert realistic_payload.get("open_used") == 16
        assert realistic_payload.get("tiers")

        rubric_fail = client.get(
            "/dev/run",
            params={
                "run": "long",
                "mode": "pass",
                "seed": 11,
                "TEST_MODE": 1,
                "TEST_GOD_MIN_NORM": 9.4,
                "TEST_GOD_MAX_SE": 0.60,
                "TEST_GOD_MIN_L2_SEEN": 12,
                "TEST_GOD_MIN_L2_ACC": 0.80,
                "TEST_GOD_MIN_OPEN": 16,
                "TEST_GOD_RUBRIC0": 0.95,
                "TEST_GOD_RUBRIC1": 0.90,
                "TEST_GOD_MIN_OPEN_RUBRIC": 0.95,
                "TEST_GOD_MIN_OPEN_RUBRIC_SEC": 0.90,
                "TEST_OPEN_FULL": 1,
            },
        )
        assert rubric_fail.status_code == 200
        rubric_payload = rubric_fail.json()
        assert any(tier != "GOD" for tier in rubric_payload.get("tiers", []))
        assert any(
            "open_rubric_low" in (flags or [])
            for flags in rubric_payload.get("reasons", {}).values()
        )

        no_force = client.get(
            "/dev/run",
            params={
                "run": "long",
                "mode": "pass",
                "seed": 12,
                "TEST_MODE": 1,
                "TEST_GOD_MIN_NORM": 0.0,
                "TEST_GOD_MAX_SE": 1.0,
                "TEST_GOD_MIN_L2_SEEN": 0,
                "TEST_GOD_MIN_L2_ACC": 0.0,
                "TEST_GOD_MIN_OPEN": 16,
            },
        )
        assert no_force.status_code == 200
        no_force_payload = no_force.json()
        assert no_force_payload.get("open_used", 0) <= 8
        assert all(tier != "GOD" for tier in no_force_payload.get("tiers", []))
        assert any(
            "open_count_low" in (flags or [])
            or "open_rubric_low" in (flags or [])
            for flags in no_force_payload.get("reasons", {}).values()
        )

        fail = client.get(
            "/dev/run",
            params={
                "run": "long",
                "mode": "pass",
                "seed": 13,
                "TEST_MODE": 1,
                "TEST_GOD_MIN_NORM": 9.9,
                "TEST_GOD_MAX_SE": 0.60,
                "TEST_GOD_MIN_L2_SEEN": 0,
                "TEST_GOD_MIN_L2_ACC": 0.65,
                "TEST_GOD_MIN_OPEN": 0,
                "TEST_OPEN_FULL": 1,
            },
        )
        assert fail.status_code == 200
        fail_payload = fail.json()
        assert all(tier != "GOD" for tier in fail_payload.get("tiers", []))
        fail_reasons = fail_payload.get("reasons", {})
        assert any("norm_low" in (vals or []) for vals in fail_reasons.values())

        leak_check = client.get("/dev/run", params={"run": "long", "mode": "pass", "seed": 14})
        assert leak_check.status_code == 200
        leak_payload = leak_check.json()
        assert all(tier != "GOD" for tier in leak_payload.get("tiers", []))

        short = client.get("/dev/run", params={"run": "short", "mode": "pass", "seed": 11})
        assert short.status_code == 200
        short_payload = short.json()
        assert short_payload.get("steps") == 40
        assert short_payload.get("tiers")
        assert all(tier == "A" for tier in short_payload["tiers"])
    finally:
        for key in [
            "STAGING_PROFILE",
            "TEST_MODE",
            "TEST_GOD_MIN_NORM",
            "TEST_GOD_MAX_SE",
            "TEST_GOD_MIN_L2_SEEN",
            "TEST_GOD_MIN_L2_ACC",
            "TEST_GOD_MIN_OPEN",
            "TEST_GOD_RUBRIC0",
            "TEST_GOD_RUBRIC1",
            "TEST_GOD_MIN_OPEN_RUBRIC",
            "TEST_GOD_MIN_OPEN_RUBRIC_SEC",
            "TEST_OPEN_FULL",
        ]:
            os.environ.pop(key, None)
        _reload_app(tmp_path)


def test_dev_run_long_obj_floor_overrides(tmp_path):
    client = _client(tmp_path)

    def _fetch(params: dict) -> dict:
        resp = client.get("/dev/run", params=params)
        assert resp.status_code == 200
        payload = resp.json()
        for key in ("steps", "sr_used", "open_used", "obj_total", "per_domain_obj_count"):
            assert key in payload, f"missing {key}"
        per_domain = payload["per_domain_obj_count"]
        assert isinstance(per_domain, dict) and len(per_domain) == 8
        assert sum(per_domain.values()) == payload["obj_total"]
        assert payload["steps"] == payload["sr_used"] + payload["open_used"] + payload["obj_total"]
        return payload

    baseline = _fetch({"run": "long", "mode": "pass", "seed": 11})
    assert baseline["obj_total"] < 112
    assert all(count >= 8 for count in baseline["per_domain_obj_count"].values())

    floor14 = _fetch(
        {
            "run": "long",
            "mode": "pass",
            "seed": 11,
            "LONG_MIN_OBJ_PER_DOMAIN": 14,
        }
    )
    assert floor14["obj_total"] == 112
    assert all(count == 14 for count in floor14["per_domain_obj_count"].values())

    floor16 = _fetch(
        {
            "run": "long",
            "mode": "pass",
            "seed": 11,
            "LONG_MIN_OBJ_PER_DOMAIN": 16,
            "LONG_ESTOP_DISABLE": 1,
        }
    )
    assert floor16["obj_total"] == 128
    assert floor16["steps"] == 152
    assert all(count == 16 for count in floor16["per_domain_obj_count"].values())
