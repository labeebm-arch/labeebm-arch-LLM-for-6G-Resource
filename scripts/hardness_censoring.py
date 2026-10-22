#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Censoring report for the random-search hardness check of Section 5.1.

Repeats the procedure of tests/test_problem.py::test_reported_random_search_difficulty
(budget 400, 15 restarts per scenario, seeds 5000 to 5014) and reports, per
scenario, the median evaluation at which random search first reached the
knowledge-based score and the number of restarts that never reached it within
the budget. A restart that never reached the score is recorded as 400, so a
scenario median of 400 is right-censored. Writes
results/reproduced/hardness_censoring.csv.
"""
from __future__ import annotations

import csv
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm6g import hard_problem as hp  # noqa: E402

BUDGET = 400
RESTARTS = 15
OUTPUT = REPO_ROOT / "results" / "reproduced" / "hardness_censoring.csv"


def main() -> None:
    rows = []
    for scenario in hp.SCENARIOS:
        informed = hp.evaluate(hp.knowledgeable_config(scenario["id"]), scenario["id"], seed=0)["score"]
        crossings, never = [], 0
        for repetition in range(RESTARTS):
            rng = random.Random(5000 + repetition)
            best, crossing = -1e9, BUDGET
            reached = False
            for evaluation in range(1, BUDGET + 1):
                best = max(best, hp.evaluate(hp.sample_random(rng), scenario["id"], seed=0)["score"])
                if best >= informed:
                    crossing, reached = evaluation, True
                    break
            crossings.append(crossing)
            never += 0 if reached else 1
        median = statistics.median(crossings)
        rows.append({
            "scenario_id": scenario["id"],
            "scenario": scenario["name"],
            "median_evaluations": median,
            "censored": "yes" if median >= BUDGET else "no",
            "restarts_never_reached": never,
            "restarts": RESTARTS,
            "crossings_sorted": " ".join(str(c) for c in sorted(crossings)),
        })
        print(f"{scenario['name']:22s} median {median:>5} never reached {never:>2}/{RESTARTS}")
    overall = statistics.median([row["median_evaluations"] for row in rows])
    total_never = sum(row["restarts_never_reached"] for row in rows)
    print(f"median of scenario medians {overall}; restarts never reached {total_never}/{RESTARTS * len(rows)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
