#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the eight-scenario proxy study for one Ollama model.

This command regenerates the one-shot and few-shot model configurations and the
five cold-search baselines. It deliberately aborts if a model response cannot
be parsed; no random fallback is used.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Dict, List

import yaml

from llm6g import hard_baselines as hb
from llm6g import hard_llm as hl
from llm6g import hard_problem as hp
from llm6g.paths import REPO_ROOT, RESULTS_DIR

METHOD_ORDER = [
    "Heuristic",
    "LLM-zeroshot",
    "LLM-fewshot",
    "GA",
    "PSO",
    "DE",
    "Bayesian",
    "Random",
]
BASELINE_NAMES = ["GA", "PSO", "DE", "Bayesian", "Random"]


def pick_examples(scenario_id: int, count: int = 3):
    others = [scenario for scenario in hp.SCENARIOS if scenario["id"] != scenario_id]
    selected = []
    selected_ids = set()
    for use_case in ["eMBB", "URLLC", "mMTC"]:
        for scenario in others:
            if scenario["use_case"] == use_case and scenario["id"] not in selected_ids:
                selected.append(scenario)
                selected_ids.add(scenario["id"])
                break
    for scenario in others:
        if len(selected) >= count:
            break
        if scenario["id"] not in selected_ids:
            selected.append(scenario)
            selected_ids.add(scenario["id"])
    return selected[:count]


def evaluations_to_match(histories: List[List[float]], target: float, budget: int) -> float:
    crossings = []
    for history in histories:
        crossing = budget + 1
        for evaluation, value in enumerate(history, start=1):
            if value >= target:
                crossing = evaluation
                break
        crossings.append(crossing)
    return st.median(crossings)


def summarize(store: Dict, scenario_ids: List[int], budget: int) -> Dict:
    summary: Dict[str, Dict] = {}
    scenario_match_medians = []
    for scenario_id in scenario_ids:
        rows = {}
        for method in METHOD_ORDER:
            scores = store[scenario_id][method]["scores"]
            best_index = max(range(len(scores)), key=scores.__getitem__)
            rows[method] = {
                "mean": st.mean(scores),
                "std": st.pstdev(scores) if len(scores) > 1 else 0.0,
                "median": st.median(scores),
                "best_score": scores[best_index],
                "best_config": store[scenario_id][method]["configs"][best_index],
            }
        fewshot_mean = rows["LLM-fewshot"]["mean"]
        transfer = {
            method: evaluations_to_match(
                store[scenario_id][method]["histories"], fewshot_mean, budget
            )
            for method in BASELINE_NAMES
        }
        scenario_match_medians.append(st.median(transfer.values()))
        scenario = hp.SCENARIO_BY_ID[scenario_id]
        summary[str(scenario_id)] = {
            "name": scenario["name"],
            "use_case": scenario["use_case"],
            "rows": rows,
            "heuristic_mean": rows["Heuristic"]["mean"],
            "zeroshot_mean": rows["LLM-zeroshot"]["mean"],
            "fewshot_mean": fewshot_mean,
            "evals_to_match_fewshot": transfer,
        }
    summary["overall"] = {
        "n_scenarios": len(scenario_ids),
        "median_transfer": st.median(scenario_match_medians),
    }
    return summary


def resolve_config_path(value: str | Path) -> Path:
    """Resolve a config path from either the working directory or repository."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    repository_candidate = REPO_ROOT / path
    return repository_candidate if repository_candidate.exists() else path


def default_output_path(config: Dict) -> Path:
    """Return a non-destructive output path for a newly executed campaign."""
    model_tag = config["ollama"]["model"].replace(":", "_").replace("/", "_")
    return RESULTS_DIR / "reproduced" / model_tag / "hard_comparison_results.json"


def run(config: Dict, output_path: Path | None = None) -> Path:
    scenario_ids = [int(value) for value in config["scenarios"]]
    budget = int(config["budget"])
    repetitions = int(config["repetitions"])
    base_seed = int(config.get("seed", 20260610))
    mock = bool(config.get("mock", False))
    ollama = config["ollama"]
    model = ollama["model"]
    url = ollama.get("url", "http://localhost:11434")

    store = {
        scenario_id: {
            method: {"scores": [], "histories": [], "configs": []}
            for method in METHOD_ORDER
        }
        for scenario_id in scenario_ids
    }

    for scenario_id in scenario_ids:
        print(f"Scenario {scenario_id}: {hp.SCENARIO_BY_ID[scenario_id]['name']}")
        for repetition in range(repetitions):
            seed = base_seed + repetition
            for name, function in hb.ALL_BASELINES.items():
                result = function(scenario_id, budget, seed)
                key = result["name"].split("(")[0]
                store[scenario_id][key]["scores"].append(result["best_score"])
                store[scenario_id][key]["histories"].append(result["history"])
                store[scenario_id][key]["configs"].append(result["best_params"])

            heuristic = hp.knowledgeable_config(scenario_id)
            heuristic_score = hp.evaluate(heuristic, scenario_id, seed=0)["score"]
            store[scenario_id]["Heuristic"]["scores"].append(heuristic_score)
            store[scenario_id]["Heuristic"]["histories"].append([heuristic_score])
            store[scenario_id]["Heuristic"]["configs"].append(heuristic)

            zero_config = hl.llm_zeroshot(scenario_id, url, model, mock, seed=seed)
            zero_score = hp.evaluate(zero_config, scenario_id, seed=0)["score"]
            store[scenario_id]["LLM-zeroshot"]["scores"].append(zero_score)
            store[scenario_id]["LLM-zeroshot"]["histories"].append([zero_score])
            store[scenario_id]["LLM-zeroshot"]["configs"].append(zero_config)

            few_config = hl.llm_fewshot(
                scenario_id,
                pick_examples(scenario_id),
                url,
                model,
                mock,
                seed=seed,
            )
            few_score = hp.evaluate(few_config, scenario_id, seed=0)["score"]
            store[scenario_id]["LLM-fewshot"]["scores"].append(few_score)
            store[scenario_id]["LLM-fewshot"]["histories"].append([few_score])
            store[scenario_id]["LLM-fewshot"]["configs"].append(few_config)
            print(f"  repetition {repetition + 1}/{repetitions}", flush=True)

    summary = summarize(store, scenario_ids, budget)
    output_path = Path(output_path) if output_path is not None else default_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "summary": summary,
        "raw": {str(key): value for key, value in store.items()},
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(f"Saved {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "proxy_study.yaml"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output JSON path. By default, new runs are written under "
            "results/reproduced/ so the immutable reference results are not overwritten."
        ),
    )
    arguments = parser.parse_args()

    config_path = resolve_config_path(arguments.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if arguments.model:
        config["ollama"]["model"] = arguments.model
    if arguments.repetitions is not None:
        config["repetitions"] = arguments.repetitions
    if arguments.budget is not None:
        config["budget"] = arguments.budget
    if arguments.scenarios:
        config["scenarios"] = [int(value) for value in arguments.scenarios.split(",")]
    if arguments.mock:
        config["mock"] = True
    run(config, Path(arguments.out) if arguments.out else None)


if __name__ == "__main__":
    main()
