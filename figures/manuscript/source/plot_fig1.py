#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure 1 of the LLM-6G manuscript: stored ns-3 histories of GA, DE and PSO
on the two earlier single-cell scenarios, 1 row x 2 columns.

Reads results/prior_ns3/figure1_histories.json (three levels up), writes
fig1_ns3_histories.csv beside this script, and saves fig1_ns3_histories.png/.pdf.
Colours and panel labels follow plot_fig3_fig5.py. No caption or title is
drawn inside the image.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PKG = Path(__file__).resolve().parents[3]
SOURCE = PKG / "results" / "prior_ns3" / "figure1_histories.json"
MARK_EVAL = 8
COLORS = {"GA": "#3775BA", "PSO": "#42949E", "DE": "#9A4D8E"}
ORDER = ["GA", "DE", "PSO"]


def style():
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def load():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    budget = int(data["budget"])
    scenarios = []
    for sid in sorted(data["scenarios"], key=int):
        entry = data["scenarios"][sid]
        methods = {m: np.asarray(entry["methods"][m], dtype=float)[:budget] for m in ORDER}
        scenarios.append((int(sid), entry["name"], methods))
    return budget, scenarios


def write_csv(budget, scenarios):
    path = HERE / "fig1_ns3_histories.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario_id", "scenario", "series", "evaluation", "score"])
        for sid, name, methods in scenarios:
            for method in ORDER:
                for k, value in enumerate(methods[method], start=1):
                    writer.writerow([sid, name, method, k, f"{value:.6f}"])
    print("Wrote", path)


def panel_label(axis, index, name):
    axis.set_title(f"({chr(96 + index)}) {name}", loc="left", fontsize=9, fontweight="bold", pad=4)


def tidy_axis(axis, values, budget, pad=0.08):
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    span = max(hi - lo, 1.0)
    axis.set_ylim(lo - pad * span, hi + pad * span)
    axis.set_xlim(0, budget)
    axis.set_xticks([0, 10, 20, 30, 40, 50, 60])
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.set_axisbelow(True)


def figure1():
    budget, scenarios = load()
    write_csv(budget, scenarios)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.05))
    x = np.arange(1, budget + 1)
    for (sid, name, methods), axis in zip(scenarios, axes):
        for method in ORDER:
            axis.plot(x, methods[method], color=COLORS[method],
                      linewidth=2.0 if method == "PSO" else 1.3)
        pso = methods["PSO"]
        at_mark, final = pso[MARK_EVAL - 1], pso[-1]
        axis.axvline(MARK_EVAL, color="#555555", linestyle="--", linewidth=0.8)
        axis.plot([MARK_EVAL], [at_mark], marker="o", markersize=4.5, color=COLORS["PSO"],
                  markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        axis.text(0.97, 0.06,
                  f"PSO at evaluation {MARK_EVAL}: {at_mark:.1f}\n({100 * at_mark / final:.0f}% of its final score)",
                  transform=axis.transAxes, fontsize=7.6, ha="right", va="bottom",
                  bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "linewidth": 0.6})
        tidy_axis(axis, np.concatenate([methods[m] for m in ORDER]), budget)
        panel_label(axis, sid, name)
        axis.set_xlabel("Simulator evaluations")
    axes[0].set_ylabel("Best-so-far score")
    handles = [Line2D([], [], color=COLORS[m], linewidth=2.0 if m == "PSO" else 1.4, label=m) for m in ORDER]
    axes[0].legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.30), ncol=1,
                   handlelength=2.0, labelspacing=0.25, borderaxespad=0.0)
    fig.tight_layout(w_pad=1.6)
    for ext in ("png", "pdf"):
        path = HERE / f"fig1_ns3_histories.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        print("Wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    style()
    figure1()
