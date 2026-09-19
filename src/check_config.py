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

    # The ordering rule that used to live here is gone, along with the defect it was
    # trying to police. research_priority() now collects every matching rule and takes
    # the LOWEST multiplier, so row order no longer determines the outcome -- it only
    # picks which name is reported among rules of equal multiplier. Requiring
    # non-increasing order was a workaround for "first match wins", and it did not work:
    # it passed while a demotion rule sat below a x1.0 rule that always claimed the same
    # papers first, so the rule never fired at all.
    #
    # What is worth checking now is a rule that can never win: identical required_groups
    # with a higher multiplier than another rule, which the lowest-wins policy will
    # always beat. Anything subtler (one rule's groups implying another's) is not
    # decidable from the config, which is why tools/check_topic_gate.py exists.
    priorities = settings.scoring.research_priorities
    for index, current in enumerate(priorities):
        for other_index, other in enumerate(priorities):
            if other_index == index or other.match_fields != current.match_fields:
                continue
            if other.required_groups == current.required_groups and other.multiplier < current.multiplier:
                problems.append(
                    f"research_priorities[{index}] '{current.name}' (x{current.multiplier}) has the "
                    f"same required_groups as '{other.name}' (x{other.multiplier}) at the same match "
                    f"scope, and the lower multiplier always wins, so this rule can never apply. "
                    f"Merge them or narrow one."
                )
                break

    names = {}
    for rule in priorities:
        if rule.name in names and names[rule.name] != rule.multiplier:
            problems.append(
                f"priority name '{rule.name}' is used with both x{names[rule.name]} and "
                f"x{rule.multiplier}; the report shows the name, so the same label would "
                f"mean two different things"
            )
        names.setdefault(rule.name, rule.multiplier)

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
