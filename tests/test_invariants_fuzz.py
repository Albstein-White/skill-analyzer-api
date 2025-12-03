"""Seed-fuzzed invariants for /dev/run staging simulator."""

from typing import Dict, List, Tuple

import pytest

import skill_core.config as cfg
from tests.test_fail_fast import _dev_run

SEEDS = range(11, 31)


@pytest.fixture(scope="module")
def invariants_tracker() -> Dict[str, List[Tuple[int, int, int, str]]]:
    return {
        "short_pass_40": [],
        "short_fail_early": [],
        "long_pass_range": [],
        "long_fail_range": [],
        "long_fail_blocks_open": [],
    }


@pytest.fixture(scope="module", autouse=True)
def _report_invariants(invariants_tracker):
    yield
    summary = {key: not value for key, value in invariants_tracker.items()}
    print("seed_invariants_summary", summary)
    for key, failures in invariants_tracker.items():
        assert not failures, f"{key} failed for seeds: {failures}"


@pytest.mark.parametrize("seed", SEEDS)
def test_dev_run_seed_invariants(tmp_path_factory, invariants_tracker, seed):
    """Ensure core fail-fast invariants hold across multiple seeds."""
    tmp_path = tmp_path_factory.mktemp(f"seed-{seed}")

    short_pass = _dev_run(tmp_path, {"run": "short", "mode": "pass", "seed": seed})
    short_fail = _dev_run(tmp_path, {"run": "short", "mode": "fail", "seed": seed})
    long_pass = _dev_run(tmp_path, {"run": "long", "mode": "pass", "seed": seed})
    long_fail = _dev_run(tmp_path, {"run": "long", "mode": "fail", "seed": seed})

    if not (short_pass.get("steps") == 40 and short_pass.get("open_used") == 0 and not short_pass.get("stop_reason")):
        invariants_tracker["short_pass_40"].append(
            (
                seed,
                short_pass.get("steps"),
                short_pass.get("open_used"),
                short_pass.get("stop_reason"),
            )
        )

    if not (
        short_fail.get("steps", 999) < 40
        and short_fail.get("stop_reason") == "fail_fast_short"
        and short_fail.get("open_used") == 0
    ):
        invariants_tracker["short_fail_early"].append(
            (
                seed,
                short_fail.get("steps"),
                short_fail.get("open_used"),
                short_fail.get("stop_reason"),
            )
        )

    long_pass_steps = long_pass.get("steps")
    if not (
        isinstance(long_pass_steps, int)
        and 100 <= long_pass_steps <= cfg.CAP_LONG
        and long_pass.get("open_used", 0) >= 8
        and not long_pass.get("stop_reason")
    ):
        invariants_tracker["long_pass_range"].append(
            (
                seed,
                long_pass_steps,
                long_pass.get("open_used"),
                long_pass.get("stop_reason"),
            )
        )

    long_fail_steps = long_fail.get("steps")
    if not (
        isinstance(long_fail_steps, int)
        and 120 <= long_fail_steps <= cfg.CAP_LONG
        and not long_fail.get("stop_reason")
    ):
        invariants_tracker["long_fail_range"].append(
            (
                seed,
                long_fail_steps,
                long_fail.get("open_used"),
                long_fail.get("stop_reason"),
            )
        )

    if not (long_fail.get("open_used") == 0):
        invariants_tracker["long_fail_blocks_open"].append(
            (
                seed,
                long_fail.get("steps"),
                long_fail.get("open_used"),
                long_fail.get("stop_reason"),
            )
        )
