import os

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
        for dom, flags in reasons.items():
            assert set(flags or []) & {"norm_low", "se_high", "open_count_low", "open_rubric_low"}, dom

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
        assert all(tier == "GOD" for tier in staging_payload["tiers"])
        assert staging_payload.get("open_used") == 16
        staging_reasons = staging_payload.get("reasons", {})
        assert all(isinstance(v, list) for v in staging_reasons.values())
        for flags in staging_reasons.values():
            assert "open_count_low" not in (flags or [])
            assert "open_rubric_low" not in (flags or [])

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
                "TEST_OPEN_FULL": 1,
            },
        )
        assert realistic.status_code == 200
        realistic_payload = realistic.json()
        assert realistic_payload.get("open_used") == 16
        assert realistic_payload.get("tiers")
        assert all(tier == "GOD" for tier in realistic_payload["tiers"])
        for flags in realistic_payload.get("reasons", {}).values():
            assert not (flags or [])

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
            "TEST_OPEN_FULL",
        ]:
            os.environ.pop(key, None)
        _reload_app(tmp_path)
