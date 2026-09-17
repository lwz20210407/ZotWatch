"""Validate the YAML configuration without touching the network.

Catches the class of mistake that previously only surfaced in a live run: weights
that no longer sum to 1.0 (which silently breaks the absolute label thresholds),
and a priority rule ordering that lets a low-multiplier exclusion shadow a
high-multiplier core rule, because the first match wins.

Run with: python -m src.check_config
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from .settings import load_settings

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHT_SUM_TOLERANCE = 1e-6


def check(base_dir: Path) -> List[str]:
    problems: List[str] = []
    settings = load_settings(base_dir)

    weights = settings.scoring.weights
    total = sum(weights.model_dump().values())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        problems.append(
            f"scoring.weights sum to {total:.4f}, not 1.0. Every component is clamped to "
            f"[0, 1], so the score only stays in [0, 1] -- and thresholds "
            f"{settings.scoring.thresholds.must_read}/{settings.scoring.thresholds.consider} "
            f"only stay meaningful -- when the weights sum to 1."
        )

    thresholds = settings.scoring.thresholds
    if thresholds.must_read <= thresholds.consider:
        problems.append("scoring.thresholds.must_read must be greater than consider")

    # research_priority() returns on the first matching rule, so multipliers must
    # be non-increasing down the list; otherwise a demoting rule placed early
    # shadows a core rule placed later.
    priorities = settings.scoring.research_priorities
    for index in range(1, len(priorities)):
        current, earlier = priorities[index], priorities[index - 1]
        if current.multiplier > earlier.multiplier:
            problems.append(
                f"research_priorities[{index}] '{current.name}' (x{current.multiplier}) ranks "
                f"higher than the preceding rule '{earlier.name}' (x{earlier.multiplier}). "
                f"First match wins, so the preceding rule shadows it. Order rules by "
                f"descending multiplier."
            )

    if settings.embedding.neighbors < 1:
        problems.append("embedding.neighbors must be at least 1")

    return problems


def main() -> int:
    problems = check(BASE_DIR)
    if problems:
        for problem in problems:
            print(f"config error: {problem}", file=sys.stderr)
        return 1
    print("Configuration OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
