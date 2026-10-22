#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sensitivity of the Section 5.6 tests to the reuse of the ten stored priors.

The 30 repetitions of the prior-guided study share ten stored few-shot
configurations (repetition r uses configuration r mod 10). This script
repeats the one-sided paired Wilcoxon test of prior-guided PSO against each
cold baseline at the full budget on (i) the 30 paired differences, as in the
manuscript, and (ii) the ten means of the three repetitions that share a
stored prior, with Holm-Bonferroni correction across the five baselines in
both cases. It also tests the full ensemble seed against the heuristic-only
seed. Numbers are printed and written to results/reproduced/
prior_reuse_sensitivity.csv.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "results" / "seeded" / "seeded30_results.json"
OUTPUT = REPO_ROOT / "results" / "reproduced" / "prior_reuse_sensitivity.csv"
BASELINES = ["GA+cold", "PSO+cold", "DE+cold", "Bayesian+cold", "Random+cold"]
PRIOR_CYCLE = 10


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def one_sided(differences: np.ndarray) -> float:
    return float(wilcoxon(differences, alternative="greater", zero_method="zsplit").pvalue)


def per_prior_means(differences: np.ndarray) -> np.ndarray:
    return np.array([differences[i::PRIOR_CYCLE].mean() for i in range(PRIOR_CYCLE)])


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    cycle = data["meta"].get("seed_cycling", "")
    assert "mod 10" in cycle, cycle
    rows = []
    for scenario_id, scenario in data["scenarios"].items():
        finals = {k: np.array([run[-1] for run in v]) for k, v in scenario["runs"].items()}
        ensemble = finals["PSO+ENS"]
        p30, p10 = [], []
        for baseline in BASELINES:
            diff = ensemble - finals[baseline]
            p30.append(one_sided(diff))
            p10.append(one_sided(per_prior_means(diff)))
        h30, h10 = holm(np.array(p30)), holm(np.array(p10))
        heur = ensemble - finals["PSO+HEUR"]
        row = {
            "scenario_id": scenario_id,
            "scenario": scenario["name"],
            "max_holm_p_n30": f"{h30.max():.3e}",
            "max_holm_p_n10": f"{h10.max():.4f}",
            "worst_baseline_n10": BASELINES[int(np.argmax(h10))],
            "ensemble_vs_heuristic_seed_p_n30": f"{one_sided(heur):.3f}",
            "ensemble_vs_heuristic_seed_p_n10": f"{one_sided(per_prior_means(heur)):.3f}",
            "ensemble_minus_heuristic_seed_mean": f"{heur.mean():+.2f}",
        }
        rows.append(row)
        print(
            f"{scenario['name']:22s} Holm max p: n=30 {row['max_holm_p_n30']}, "
            f"n=10 {row['max_holm_p_n10']} ({row['worst_baseline_n10']}); "
            f"ensemble vs heuristic seed p: n=30 {row['ensemble_vs_heuristic_seed_p_n30']}, "
            f"n=10 {row['ensemble_vs_heuristic_seed_p_n10']}, mean {row['ensemble_minus_heuristic_seed_mean']}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
