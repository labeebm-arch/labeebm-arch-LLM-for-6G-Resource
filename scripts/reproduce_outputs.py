#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate manuscript tables and data-driven figures from stored results."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm6g import hard_problem as hp
from llm6g.paths import FIGURES_DIR, RESULTS_DIR, TABLES_DIR
from llm6g.stored_results import (
    LLAMA_DIR,
    QWEN_DIR,
    SEEDED_RESULTS_DIR,
    load_json,
    load_model_campaign,
)

BASELINES = ["GA", "PSO", "DE", "Bayesian", "Random"]


def write_csv(path: Path, header: Iterable, rows: Iterable[Iterable]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)
    print(f"Wrote {path}")


def mean_curve(histories: List[List[float]], budget: int = 60) -> np.ndarray:
    padded = []
    for history in histories:
        if not history:
            continue
        padded.append((list(history) + [history[-1]] * budget)[:budget])
    return np.mean(np.asarray(padded, dtype=float), axis=0)


def format_match(value: float, budget: int = 60):
    return f">{budget}" if value > budget else value


def table3(llama: Dict, qwen: Dict) -> None:
    rows = []
    for scenario_id in range(1, 9):
        sid = str(scenario_id)
        lsum = llama["summary"][sid]
        qsum = qwen["summary"][sid]
        lmatch = st.median(lsum["evals_to_match_fewshot"].values())
        qmatch = st.median(qsum["evals_to_match_fewshot"].values())
        rows.append([
            lsum["name"],
            round(lsum["heuristic_mean"], 1),
            round(lsum["zeroshot_mean"], 1),
            round(lsum["fewshot_mean"], 1),
            format_match(lmatch),
            round(qsum["zeroshot_mean"], 1),
            round(qsum["fewshot_mean"], 1),
            format_match(qmatch),
        ])
    write_csv(
        TABLES_DIR / "table3_one_shot_and_sample_efficiency.csv",
        [
            "Scenario", "Heuristic", "Llama zero-shot", "Llama few-shot",
            "Llama match", "Qwen zero-shot", "Qwen few-shot", "Qwen match",
        ],
        rows,
    )


def table4(seeded: Dict) -> None:
    rows = []
    for scenario_id in range(1, 9):
        data = seeded["scenarios"][str(scenario_id)]
        finals = [history[-1] for history in data["runs"]["PSO+ENS"]]
        stats = data["stats"]
        rows.append([
            data["name"],
            round(data["heuristic"], 1),
            round(stats["best_cold_final"], 1),
            round(st.mean(finals), 1),
            round(st.pstdev(finals), 1),
            stats["evals_to_pass_best_cold"],
            max(stats["holm_p_vs_cold"].values()),
            round(100 * stats["anytime_vs_envelope"]),
            min(stats["cliffs_vs_cold"].values()),
        ])
    write_csv(
        TABLES_DIR / "table4_prior_guided_pso.csv",
        [
            "Scenario", "Heuristic", "Best cold", "Prior-guided PSO mean",
            "Prior-guided PSO std", "Pass evaluation", "Worst Holm p",
            "Anytime share (%)", "Minimum Cliff delta",
        ],
        rows,
    )


def table5(seeded: Dict) -> None:
    rows = []
    for scenario_id in range(1, 9):
        data = seeded["scenarios"][str(scenario_id)]
        means = {
            method: st.mean(history[-1] for history in data["runs"][method])
            for method in ["PSO+ENS", "GA+ENS", "DE+ENS", "Bayesian+ENS"]
        }
        rows.append([
            data["name"],
            round(means["PSO+ENS"], 1),
            round(means["GA+ENS"], 1),
            round(means["DE+ENS"], 1),
            round(means["Bayesian+ENS"], 1),
        ])
    write_csv(
        TABLES_DIR / "table5_host_ablation.csv",
        ["Scenario", "PSO", "GA", "DE", "Bayesian"],
        rows,
    )


def table6(seeded: Dict) -> None:
    rows = []
    for scenario_id in range(1, 9):
        data = seeded["scenarios"][str(scenario_id)]
        runs = data["runs"]
        mean_final = lambda method: st.mean(history[-1] for history in runs[method])
        rows.append([
            data["name"],
            round(data["stats"]["best_cold_final"], 1),
            round(mean_final("PSO+L"), 1),
            round(mean_final("PSO+Q"), 1),
            round(mean_final("PSO+HEUR"), 1),
            round(mean_final("PSO+ENS"), 1),
        ])
    write_csv(
        TABLES_DIR / "table6_seed_ablation.csv",
        ["Scenario", "Best cold", "Llama-only", "Qwen-only", "Heuristic-only", "Ensemble"],
        rows,
    )


def table7() -> None:
    validation = load_json(RESULTS_DIR / "ns3" / "ns3_validation.json")
    oneshot = load_json(RESULTS_DIR / "ns3" / "ns3_oneshot.json")
    calibration = load_json(RESULTS_DIR / "ns3" / "calibration_summary.json")
    rows = []
    for scenario_id in [1, 2, 4]:
        sid = str(scenario_id)
        comparison = validation["comparison"][sid]
        rows.append([
            comparison["name"],
            calibration["spearman_rho"][sid],
            round(comparison["heuristic"], 1),
            round(oneshot[sid]["best_oneshot_score"], 1),
            round(comparison["best_cold_final"], 1),
            round(comparison["prior_guided_final"], 1),
        ])
    write_csv(
        TABLES_DIR / "table7_packet_level_validation.csv",
        ["Scenario", "Calibration rho", "Heuristic", "One-shot", "Best cold", "Prior-guided"],
        rows,
    )


def appendix_a(llama: Dict) -> None:
    rows = []
    for scenario_id in range(1, 9):
        summary = llama["summary"][str(scenario_id)]
        values = [summary["name"], round(summary["heuristic_mean"], 1)]
        for method in BASELINES:
            row = summary["rows"][method]
            values.extend([round(row["mean"], 1), round(row["std"], 1)])
        rows.append(values)
    header = ["Scenario", "Heuristic"]
    for method in BASELINES:
        header.extend([f"{method} mean", f"{method} std"])
    write_csv(TABLES_DIR / "appendix_A1_cold_baselines.csv", header, rows)


def appendix_b(model: Dict, model_name: str) -> None:
    rows = []
    for scenario_id in range(1, 9):
        summary = model["summary"][str(scenario_id)]
        transfer = summary["evals_to_match_fewshot"]
        rows.append([summary["name"]] + [format_match(transfer[name]) for name in BASELINES])
    write_csv(
        TABLES_DIR / f"appendix_B1_{model_name}_transfer.csv",
        ["Scenario"] + BASELINES,
        rows,
    )


def appendix_c() -> None:
    data = load_json(RESULTS_DIR / "model_campaigns" / "best_configurations_table_c.json")
    scenario_ids = [str(value) for value in data["scenario_order"]]
    rows = []
    for parameter in hp.PARAM_ORDER:
        rows.append([parameter] + [data["configurations"][sid][parameter] for sid in scenario_ids])
    write_csv(
        TABLES_DIR / "appendix_C1_best_configurations.csv",
        ["Parameter"] + [f"S{sid}" for sid in scenario_ids],
        rows,
    )



def plot_figure1(output_dir: Path) -> None:
    """Regenerate the historical two-panel convergence figure used as Fig. 1."""
    data = load_json(RESULTS_DIR / "prior_ns3" / "figure1_histories.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for axis, scenario_id in zip(axes, ["1", "2"]):
        scenario = data["scenarios"][scenario_id]
        for method in ["GA", "DE", "PSO"]:
            curve = scenario["methods"][method]
            axis.plot(np.arange(1, len(curve) + 1), curve, label=method, linewidth=1.4)
        pso_curve = scenario["methods"]["PSO"]
        eval8 = pso_curve[7]
        fraction = 100.0 * eval8 / pso_curve[-1]
        axis.axvline(8, linestyle=":", linewidth=1)
        axis.annotate(
            f"PSO at eval 8: {eval8:.1f}\n({fraction:.0f}% of its final score)",
            xy=(8, eval8),
            xytext=(13, min(99.0, eval8 + 1.5)),
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            fontsize=8,
        )
        axis.set_title(scenario["name"], fontsize=10)
        axis.set_xlabel("Simulation evaluations")
        axis.set_xlim(1, data["budget"])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Best-so-far score")
    fig.tight_layout()
    path = output_dir / "figure1_prior_ns3_histories_reproduced.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Wrote {path}")

def plot_figure3(llama: Dict, qwen: Dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True)
    for scenario_id, axis in enumerate(axes.flat, start=1):
        sid = str(scenario_id)
        raw = llama["raw"][sid]
        for method in BASELINES:
            axis.plot(np.arange(1, 61), mean_curve(raw[method]["histories"]), label=method, linewidth=1.2)
        axis.axhline(llama["summary"][sid]["heuristic_mean"], linestyle="-", linewidth=1.2, label="Expert heuristic")
        axis.axhline(llama["summary"][sid]["fewshot_mean"], linestyle="--", linewidth=1.2, label="Llama few-shot")
        axis.axhline(qwen["summary"][sid]["fewshot_mean"], linestyle=":", linewidth=1.4, label="Qwen few-shot")
        axis.set_title(f"{chr(96 + scenario_id)}) {llama['summary'][sid]['name']}", fontsize=9)
        axis.set_xlim(1, 60)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Evaluations")
    axes[1, 1].set_xlabel("Evaluations")
    axes[1, 2].set_xlabel("Evaluations")
    axes[1, 3].set_xlabel("Evaluations")
    axes[0, 0].set_ylabel("Best-so-far score")
    axes[1, 0].set_ylabel("Best-so-far score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = output_dir / "figure3_convergence_reproduced.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_figure4(llama: Dict, qwen: Dict, output_dir: Path) -> None:
    labels, llama_values, qwen_values = [], [], []
    for scenario_id in range(1, 9):
        sid = str(scenario_id)
        labels.append(llama["summary"][sid]["name"])
        llama_values.append(st.median(llama["summary"][sid]["evals_to_match_fewshot"].values()))
        qwen_values.append(st.median(qwen["summary"][sid]["evals_to_match_fewshot"].values()))
    x = np.arange(len(labels))
    width = 0.38
    fig, axis = plt.subplots(figsize=(11, 5.2))
    bars_l = axis.bar(x - width / 2, np.minimum(llama_values, 61), width, label="Llama-3-8B few-shot")
    bars_q = axis.bar(x + width / 2, np.minimum(qwen_values, 61), width, label="Qwen2.5-7B few-shot")
    axis.axhline(60, linestyle="--", linewidth=1)
    axis.set_ylabel("Median evaluations to match")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylim(0, 65)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for bars, values in [(bars_l, llama_values), (bars_q, qwen_values)]:
        for bar, value in zip(bars, values):
            if value > 60:
                axis.text(bar.get_x() + bar.get_width() / 2, 61.2, ">60", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = output_dir / "figure4_match_evaluations_reproduced.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_figure5(seeded: Dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True)
    for scenario_id, axis in enumerate(axes.flat, start=1):
        data = seeded["scenarios"][str(scenario_id)]
        for method in BASELINES:
            axis.plot(
                np.arange(1, 61),
                mean_curve(data["runs"][f"{method}+cold"]),
                label=method,
                linewidth=1.1,
            )
        axis.plot(
            np.arange(1, 61),
            mean_curve(data["runs"]["PSO+ENS"]),
            label="Prior-guided PSO",
            linewidth=1.8,
        )
        axis.axhline(data["heuristic"], linestyle="-", linewidth=1.2, label="Expert heuristic")
        pass_eval = data["stats"]["evals_to_pass_best_cold"]
        if pass_eval <= 60:
            axis.axvline(pass_eval, linestyle=":", linewidth=1)
        axis.set_title(f"{chr(96 + scenario_id)}) {data['name']}", fontsize=9)
        axis.set_xlim(1, 60)
        axis.grid(alpha=0.25)
    for axis in axes[1, :]:
        axis.set_xlabel("Evaluations")
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = output_dir / "figure5_prior_guided_reproduced.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate manuscript tables and data-driven figures from stored results."
    )
    parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    generated_figures = FIGURES_DIR / "generated"
    generated_figures.mkdir(parents=True, exist_ok=True)
    llama = load_model_campaign(LLAMA_DIR)
    qwen = load_model_campaign(QWEN_DIR)
    seeded = load_json(SEEDED_RESULTS_DIR / "seeded30_results.json")

    table3(llama, qwen)
    table4(seeded)
    table5(seeded)
    table6(seeded)
    table7()
    appendix_a(llama)
    appendix_b(llama, "llama")
    appendix_b(qwen, "qwen")
    appendix_c()
    plot_figure1(generated_figures)
    plot_figure3(llama, qwen, generated_figures)
    plot_figure4(llama, qwen, generated_figures)
    plot_figure5(seeded, generated_figures)


if __name__ == "__main__":
    main()
