"""Validate the YAML configuration without touching the network.

Catches the class of mistake that previously only surfaced in a live run: weights
that no longer sum to 1.0 (which silently breaks the absolute label thresholds),
and a priority rule ordering that lets a low-multiplier exclusion shadow a
high-multiplier core rule, because the first match wins.

Run with: python -m src.check_config
"""
from __future__ import annotations

import os
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

    # research_priority() returns on the first matching rule, so multipliers must be
    # non-increasing down the list; otherwise a demoting rule placed early shadows a
    # core rule placed later.
    #
    # Compared within a match scope, not across. A title-scoped rule reads a strict
    # subset of what a title_abstract rule reads, so putting a low-multiplier
    # title rule above a higher title_abstract rule is a deliberate narrowing: it can
    # only fire when the marker is in the title itself. That is how 表征主导 (x0.55,
    # title) sits above 机制参考 (x0.85) -- a paper merely using EBSD in its
    # fractography still reaches 机制参考; one titled "EBSD characterization of ..."
    # does not.
    priorities = settings.scoring.research_priorities
    for index, current in enumerate(priorities):
        for earlier in priorities[:index]:
            if earlier.match_fields != current.match_fields:
                continue
            if current.multiplier > earlier.multiplier:
                problems.append(
                    f"research_priorities[{index}] '{current.name}' (x{current.multiplier}) ranks "
                    f"higher than the preceding rule '{earlier.name}' (x{earlier.multiplier}) at "
                    f"the same match scope ({current.match_fields}). First match wins, so the "
                    f"preceding rule shadows it. Order rules by descending multiplier."
                )
                break

    if settings.embedding.neighbors < 1:
        problems.append("embedding.neighbors must be at least 1")

    embedding = settings.embedding
    if embedding.provider != "local":
        if not embedding.base_url.startswith("http"):
            problems.append(f"embedding.base_url must be an http(s) URL, got {embedding.base_url!r}")
        if not embedding.api_key_env:
            problems.append("embedding.api_key_env must name the environment variable holding the key")
        if not os.getenv(embedding.api_key_env) and not embedding.local_fallback_model:
            problems.append(
                f"{embedding.api_key_env} is unset and no local_fallback_model is configured, "
                f"so embedding would fail at run time."
            )

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
