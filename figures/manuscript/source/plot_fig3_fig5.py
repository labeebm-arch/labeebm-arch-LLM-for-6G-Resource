#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figures 3 and 5 of the LLM-6G manuscript, 4 rows x 2 columns, full page.

Reads the CSV files written by extract_fig_data.py (same folder) and writes
fig3_convergence.png/.pdf and fig5_prior_guided.png/.pdf. No caption or
title is drawn inside the image; panels carry a short label only.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
BUDGET = 60

# Colour roles. Cold baselines keep one colour each across both figures.
# Llama and Qwen keep the orange and green used in Figure 4.
COLORS = {
    "GA": "#3775BA",
    "PSO": "#42949E",
    "DE": "#9A4D8E",
    "Bayesian": "#B64342",
    "Random": "#8C8C8C",
    "Prior-guided PSO": "#0F4D92",
    "Expert heuristic": "#000000",
    "Llama-3-8B few-shot": "#E07B00",
    "Qwen2.5-7B few-shot": "#2E8B57",
}
BASELINES = ["GA", "PSO", "DE", "Bayesian", "Random"]


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


def read_curves(path):
    curves = defaultdict(dict)  # curves[sid][series] -> np.array of 60 scores
    names = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sid = int(row["scenario_id"])
            names[sid] = row["scenario"]
            curves[sid].setdefault(row["series"], [None] * BUDGET)
            curves[sid][row["series"]][int(row["evaluation"]) - 1] = float(row["score"])
    for sid in curves:
        for series in curves[sid]:
            curves[sid][series] = np.asarray(curves[sid][series], dtype=float)
    return curves, names


def read_refs(path):
    refs = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            refs[int(row["scenario_id"])] = row
    return refs


def panel_label(axis, index, name):
    axis.set_title(f"({chr(96 + index)}) {name}", loc="left", fontsize=9, fontweight="bold", pad=4)


def tidy_axis(axis, values, pad=0.08):
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    span = max(hi - lo, 1.0)
    axis.set_ylim(lo - pad * span, hi + pad * span)
    axis.set_xlim(0, BUDGET)
    axis.set_xticks([0, 10, 20, 30, 40, 50, 60])
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.set_axisbelow(True)


def new_page_figure():
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 10.4), sharex=True)
    return fig, axes.flat


def save(fig, stem):
    for ext in ("png", "pdf"):
        path = HERE / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        print("Wrote", path)
    plt.close(fig)


def figure3():
    curves, names = read_curves(HERE / "fig3_convergence.csv")
    refs = read_refs(HERE / "fig3_references.csv")
    fig, axes = new_page_figure()
    x = np.arange(1, BUDGET + 1)
    for sid, axis in enumerate(axes, start=1):
        seen = []
        for method in BASELINES:
            y = curves[sid][method]
            axis.plot(x, y, color=COLORS[method], linewidth=1.3)
            seen.append(y)
        r = refs[sid]
        h, l, q = float(r["heuristic"]), float(r["llama_fewshot"]), float(r["qwen_fewshot"])
        axis.axhline(h, color=COLORS["Expert heuristic"], linestyle="-", linewidth=1.1)
        axis.axhline(l, color=COLORS["Llama-3-8B few-shot"], linestyle="--", linewidth=1.3)
        axis.axhline(q, color=COLORS["Qwen2.5-7B few-shot"], linestyle=(0, (3, 1.5, 1, 1.5)), linewidth=1.3)
        tidy_axis(axis, np.concatenate(seen + [np.array([h, l, q])]))
        panel_label(axis, sid, names[sid])
        if sid % 2 == 1:
            axis.set_ylabel("Best-so-far score")
        if sid >= 7:
            axis.set_xlabel("Simulator evaluations")
    handles = [Line2D([], [], color=COLORS[m], linewidth=1.6, label=m) for m in BASELINES]
    handles += [
        Line2D([], [], color=COLORS["Expert heuristic"], linestyle="-", linewidth=1.3, label="Expert heuristic"),
        Line2D([], [], color=COLORS["Llama-3-8B few-shot"], linestyle="--", linewidth=1.5, label="Llama-3-8B few-shot"),
        Line2D([], [], color=COLORS["Qwen2.5-7B few-shot"], linestyle=(0, (3, 1.5, 1, 1.5)), linewidth=1.5, label="Qwen2.5-7B few-shot"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.2, w_pad=1.5)
    save(fig, "fig3_convergence")


def figure5():
    curves, names = read_curves(HERE / "fig5_prior_guided.csv")
    refs = read_refs(HERE / "fig5_references.csv")
    fig, axes = new_page_figure()
    x = np.arange(1, BUDGET + 1)
    for sid, axis in enumerate(axes, start=1):
        seen = []
        for method in BASELINES:
            y = curves[sid][method]
            axis.plot(x, y, color=COLORS[method], linewidth=1.1, alpha=0.9)
            seen.append(y)
        y = curves[sid]["Prior-guided PSO"]
        axis.plot(x, y, color=COLORS["Prior-guided PSO"], linewidth=2.3)
        seen.append(y)
        r = refs[sid]
        h, best_cold, pass_eval = float(r["heuristic"]), float(r["best_cold_final"]), int(r["pass_eval"])
        axis.axhline(h, color=COLORS["Expert heuristic"], linestyle=":", linewidth=1.1)
        if pass_eval <= BUDGET:
            axis.axvline(pass_eval, color=COLORS["Prior-guided PSO"], linestyle="--", linewidth=0.9, alpha=0.8)
            axis.plot([pass_eval], [y[pass_eval - 1]], marker="o", markersize=5,
                      color=COLORS["Prior-guided PSO"], markeredgecolor="white", markeredgewidth=0.8, zorder=5)
            axis.text(0.98, 0.04, f"passes best cold final ({best_cold:.1f}) at evaluation {pass_eval}",
                      transform=axis.transAxes, ha="right", va="bottom", fontsize=7.6,
                      bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "linewidth": 0.6})
        tidy_axis(axis, np.concatenate(seen + [np.array([h])]))
        panel_label(axis, sid, names[sid])
        if sid % 2 == 1:
            axis.set_ylabel("Best-so-far score")
        if sid >= 7:
            axis.set_xlabel("Simulator evaluations")
    handles = [Line2D([], [], color=COLORS[m], linewidth=1.4, label=m) for m in BASELINES]
    handles += [
        Line2D([], [], color=COLORS["Prior-guided PSO"], linewidth=2.3, label="Prior-guided PSO (this work)"),
        Line2D([], [], color=COLORS["Expert heuristic"], linestyle=":", linewidth=1.3, label="Expert heuristic"),
        Line2D([], [], color=COLORS["Prior-guided PSO"], linestyle="--", linewidth=1.0, marker="o", markersize=5,
               markeredgecolor="white", label="Pass of best cold final"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.2, w_pad=1.5)
    save(fig, "fig5_prior_guided")


if __name__ == "__main__":
    style()
    figure3()
    figure5()
