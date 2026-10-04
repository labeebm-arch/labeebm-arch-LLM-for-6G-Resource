#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Extract the data behind Figures 3 and 5 from the stored result JSON files.

Source package: LLM6G_GitHub_Ready_v1.0.3 (results/model_campaigns and results/seeded).
Outputs (same folder as this script):
  fig3_convergence.csv   long format: scenario_id, scenario, series, evaluation, score
  fig3_references.csv    per scenario: heuristic, llama_fewshot, qwen_fewshot
  fig5_prior_guided.csv  long format: scenario_id, scenario, series, evaluation, score
  fig5_references.csv    per scenario: heuristic, best_cold_final, pass_eval
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[3]  # repository root
OUT = Path(__file__).resolve().parent
BASELINES = ["GA", "PSO", "DE", "Bayesian", "Random"]
BUDGET = 60


def mean_curve(histories, budget=BUDGET):
    padded = []
    for history in histories:
        if not history:
            continue
        padded.append((list(history) + [history[-1]] * budget)[:budget])
    return np.mean(np.asarray(padded, dtype=float), axis=0)


def write(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print("Wrote", path, len(rows), "rows")


def main():
    llama = json.load(open(PKG / "results/model_campaigns/llama3_8b/hard_comparison_results.json"))
    qwen = json.load(open(PKG / "results/model_campaigns/qwen2.5_7b-instruct-q5_K_M/hard_comparison_results.json"))
    seeded = json.load(open(PKG / "results/seeded/seeded30_results.json"))

    # Figure 3: cold baselines mean best-so-far curves plus three horizontal references
    rows, refs = [], []
    for sid in range(1, 9):
        s = str(sid)
        name = llama["summary"][s]["name"]
        for method in BASELINES:
            curve = mean_curve(llama["raw"][s][method]["histories"])
            for k, v in enumerate(curve, start=1):
                rows.append([sid, name, method, k, round(float(v), 4)])
        refs.append([
            sid, name,
            round(llama["summary"][s]["heuristic_mean"], 4),
            round(llama["summary"][s]["fewshot_mean"], 4),
            round(qwen["summary"][s]["fewshot_mean"], 4),
        ])
    write(OUT / "fig3_convergence.csv", ["scenario_id", "scenario", "series", "evaluation", "score"], rows)
    write(OUT / "fig3_references.csv", ["scenario_id", "scenario", "heuristic", "llama_fewshot", "qwen_fewshot"], refs)

    # Figure 5: cold baselines and prior-guided PSO (30 repetitions, mean curves)
    rows, refs = [], []
    for sid in range(1, 9):
        s = str(sid)
        data = seeded["scenarios"][s]
        name = data["name"]
        for method in BASELINES:
            curve = mean_curve(data["runs"][f"{method}+cold"])
            for k, v in enumerate(curve, start=1):
                rows.append([sid, name, method, k, round(float(v), 4)])
        curve = mean_curve(data["runs"]["PSO+ENS"])
        for k, v in enumerate(curve, start=1):
            rows.append([sid, name, "Prior-guided PSO", k, round(float(v), 4)])
        refs.append([
            sid, name,
            round(data["heuristic"], 4),
            round(data["stats"]["best_cold_final"], 4),
            data["stats"]["evals_to_pass_best_cold"],
        ])
    write(OUT / "fig5_prior_guided.csv", ["scenario_id", "scenario", "series", "evaluation", "score"], rows)
    write(OUT / "fig5_references.csv", ["scenario_id", "scenario", "heuristic", "best_cold_final", "pass_eval"], refs)


if __name__ == "__main__":
    main()
