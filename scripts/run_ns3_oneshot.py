# SPDX-License-Identifier: MIT
"""One-shot / few-shot LLM configuration quality in NS-3 (5G-LENA).

Evaluates, directly in the packet simulator and with no search, the LLM
zero-shot and few-shot configurations, the expert heuristic, and a set of random
configurations, for each scenario. This isolates the value of the prior itself:
the configuration an engineer would deploy after one or a few model queries,
before spending any expensive simulation budget.

Output: `results/reproduced/ns3_oneshot.json` by default, plus a printed
per-scenario summary.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from llm6g import hard_problem as hp
from llm6g import ns3_problem as n3
from llm6g.paths import RESULTS_DIR
from llm6g.stored_results import best_campaign_config

LLAMA = "llama3_8b"
QWEN = "qwen2.5_7b-instruct-q5_K_M"
DEFAULT_SCENARIOS = [1, 2, 4]
DEFAULT_N_RANDOM = 6


def load_cfg(sid: int, model_dir: str, key: str):
    """Load a stored parsed model configuration."""
    return best_campaign_config(model_dir, sid, key)


def build_tasks(scenarios, n_random: int):
    rng = np.random.default_rng(7)
    tasks = []
    for sid in scenarios:
        tasks.append((sid, "heuristic", hp.knowledgeable_config(sid)))
        for md, tag in [(LLAMA, "llama"), (QWEN, "qwen")]:
            z = load_cfg(sid, md, "LLM-zeroshot")
            f = load_cfg(sid, md, "LLM-fewshot")
            if z is not None:
                tasks.append((sid, "zeroshot_%s" % tag, z))
            if f is not None:
                tasks.append((sid, "fewshot_%s" % tag, f))
        for i in range(n_random):
            tasks.append((sid, "random%d" % i, hp.decode_vector(rng.random(n3.DIM))))
    return tasks


def _work(task):
    sid, tag, cfg = task
    return (sid, tag, float(n3.evaluate(cfg, sid)["score"]))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stored one-shot and few-shot priors in ns-3/5G-LENA."
    )
    parser.add_argument("--scenarios", default="1,2,4")
    parser.add_argument("--random-configs", type=int, default=DEFAULT_N_RANDOM)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", "12")))
    parser.add_argument(
        "--out",
        default=str(RESULTS_DIR / "reproduced" / "ns3_oneshot.json"),
    )
    arguments = parser.parse_args()
    scenarios = [int(value) for value in arguments.scenarios.split(",") if value.strip()]
    unknown = [sid for sid in scenarios if sid not in hp.SCENARIO_BY_ID]
    if unknown:
        parser.error(f"unknown scenario IDs: {unknown}")
    if arguments.random_configs < 1:
        parser.error("--random-configs must be at least 1")
    if arguments.jobs < 1:
        parser.error("--jobs must be at least 1")

    n3.ensure_available()
    tasks = build_tasks(scenarios, arguments.random_configs)
    print("Evaluating %d (scenario, config) pairs in NS-3 on %d workers ..." % (len(tasks), arguments.jobs))
    if arguments.jobs == 1:
        outs = list(map(_work, tasks))
    else:
        with mp.Pool(arguments.jobs) as pool:
            outs = pool.map(_work, tasks)

    by = {sid: {} for sid in scenarios}
    for sid, tag, score in outs:
        by[sid][tag] = score

    res = {}
    for sid in scenarios:
        d = by[sid]
        rand = [v for k, v in d.items() if k.startswith("random")]
        oneshot = {k: v for k, v in d.items() if k.startswith(("zeroshot", "fewshot"))}
        best_os = max(oneshot, key=oneshot.get)
        res[str(sid)] = {
            "name": hp.SCENARIO_BY_ID[sid]["name"],
            "heuristic": d["heuristic"],
            "random_mean": float(np.mean(rand)), "random_max": float(np.max(rand)),
            "oneshot": oneshot,
            "best_oneshot": best_os, "best_oneshot_score": oneshot[best_os],
        }
        print("\n### scenario %d: %s" % (sid, res[str(sid)]["name"]))
        print("   random: mean=%.1f  max=%.1f" % (res[str(sid)]["random_mean"], res[str(sid)]["random_max"]))
        print("   heuristic            = %.1f" % d["heuristic"])
        for k, v in sorted(oneshot.items(), key=lambda x: -x[1]):
            print("   %-20s = %.1f" % (k, v))
        print("   best one-shot: %s = %.1f  (vs heuristic %.1f, random-mean %.1f)"
              % (best_os, oneshot[best_os], d["heuristic"], res[str(sid)]["random_mean"]))

    output_path = Path(arguments.out).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(res, handle, indent=1)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(output_path)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
