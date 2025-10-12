from __future__ import annotations

import pytest

from skill_core.question_bank import DOMAINS
from skill_core.types import Item


def build_synthetic_bank(
    *,
    domains: list[str] | None = None,
    mcq_per_level: int = 3,
    include_sr: bool = True,
    include_open: bool = True,
) -> list[Item]:
    """Create a deterministic synthetic bank for tests and smoke runs."""

    items: list[Item] = []
    target_domains = domains or list(DOMAINS)
    for domain in target_domains:
        for level in range(-2, 3):
            for idx in range(mcq_per_level):
                items.append(
                    Item(
                        id=f"{domain}_mcq_{level}_{idx}",
                        domain=domain,
                        type="MCQ",
                        text=f"{domain} MCQ {level} #{idx}",
                        options=["A", "B", "C", "D"],
                        correct=0,
                        difficulty=level,
                        variant_group=f"{domain}_mcq_vg_{level}_{idx}",
                    )
                )

        if include_sr:
            for idx in range(2):
                items.append(
                    Item(
                        id=f"{domain}_sr_{idx}",
                        domain=domain,
                        type="SR",
                        text=f"Rate your {domain} confidence",
                        options=["0", "1", "2", "3", "4"],
                        difficulty=0,
                        variant_group=f"{domain}_sr_vg_{idx}",
                    )
                )

        if include_open:
            for level in (-1, 0, 1):
                items.append(
                    Item(
                        id=f"{domain}_open_{level}",
                        domain=domain,
                        type="OPEN",
                        text=f"Describe a {domain} challenge",
                        difficulty=level,
                        variant_group=f"{domain}_open_vg_{level}",
                    )
                )

    return items


@pytest.fixture
def synthetic_bank() -> list[Item]:
    return build_synthetic_bank()

