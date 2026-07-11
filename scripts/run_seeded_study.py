#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reproduce the 30-repetition seeded study used in Tables 4--6."""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Dict

from scipy.stats import wilcoxon

from llm6g import hard_baselines as hb
from llm6g import hard_problem as hp
from llm6g.paths import REPO_ROOT, RESULTS_DIR
from llm6g.seeded import (
    SEED_HOSTS,
    cliffs_delta,
    holm_bonferroni,
    mean_curve,
    pso_seeded,
)
from llm6g.stored_results import (
    LLAMA_DIR,
    MODEL_CONFIG_CYCLE,
    QWEN_DIR,
    load_json,
    seed_bank_path,
    sha256_file,
)

BASELINE_NAMES = ["GA", "PSO", "DE", "Bayesian", "Random"]
SEED_SETS = {
    "PSO+L": ["llama"],
    "PSO+Q": ["qwen"],
    "PSO+HEUR": ["heuristic"],
}


def _stored_config_from_payload(
    payload: Dict, scenario_id: int, method: str, repetition: int, source_name: str
) -> Dict:
    configs = payload["raw"][str(scenario_id)][method]["configs"]
    if len(configs) < MODEL_CONFIG_CYCLE:
        raise ValueError(
            f"{source_name}, scenario {scenario_id}, {method} has only "
            f"{len(configs)} configurations; at least {MODEL_CONFIG_CYCLE} are required."
        )
    return hp.validate(configs[repetition % MODEL_CONFIG_CYCLE])



def _portable_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def run(arguments: argparse.Namespace) -> Dict:
    llama_path = Path(arguments.llama_results).expanduser().resolve()
    qwen_path = Path(arguments.qwen_results).expanduser().resolve()
    llama_payload = load_json(llama_path)
    qwen_payload = load_json(qwen_path)

    output = {
        "scenarios": {},
        "meta": {
            "budget": arguments.budget,
            "reps": arguments.reps,
            "base_seed": arguments.base_seed,
            "seed_cycling": "rep r uses stored config r mod 10",
            "pairing": "all methods share seed BASE_SEED+r",
            "llama_results": _portable_path(llama_path),
            "llama_results_sha256": sha256_file(llama_path),
            "llama_source_repetitions": llama_payload.get("config", {}).get("repetitions"),
            "qwen_results": _portable_path(qwen_path),
            "qwen_results_sha256": sha256_file(qwen_path),
            "qwen_source_repetitions": qwen_payload.get("config", {}).get("repetitions"),
        },
    }

    for scenario_id in arguments.scenarios:
        scenario = hp.SCENARIO_BY_ID[scenario_id]
        runs: Dict[str, list] = {}
        for name, function in hb.ALL_BASELINES.items():
            runs[f"{name}+cold"] = [
                function(scenario_id, arguments.budget, arguments.base_seed + repetition)[
                    "history"
                ]
                for repetition in range(arguments.reps)
            ]

        for repetition in range(arguments.reps):
            llama = _stored_config_from_payload(
                llama_payload, scenario_id, "LLM-fewshot", repetition, str(llama_path)
            )
            qwen = _stored_config_from_payload(
                qwen_payload, scenario_id, "LLM-fewshot", repetition, str(qwen_path)
            )
            heuristic = hp.knowledgeable_config(scenario_id)
            ensemble = [qwen, llama, heuristic]
            seed = arguments.base_seed + repetition

            for tag, function in SEED_HOSTS.items():
                runs.setdefault(tag, []).append(
                    function(scenario_id, arguments.budget, seed, ensemble)
                )
            sources = {"llama": llama, "qwen": qwen, "heuristic": heuristic}
            for tag, source_names in SEED_SETS.items():
                configs = [sources[name] for name in source_names]
                runs.setdefault(tag, []).append(
                    pso_seeded(scenario_id, arguments.budget, seed, configs)
                )

        cold = {
            key.removesuffix("+cold"): [history[-1] for history in values]
            for key, values in runs.items()
            if key.endswith("+cold")
        }
        ensemble_finals = [history[-1] for history in runs["PSO+ENS"]]
        raw_p = {
            method: float(
                wilcoxon(
                    ensemble_finals,
                    scores,
                    alternative="greater",
                    zero_method="zsplit",
                ).pvalue
            )
            for method, scores in cold.items()
        }
        stats = {
            "holm_p_vs_cold": holm_bonferroni(raw_p),
            "cliffs_vs_cold": {
                method: cliffs_delta(ensemble_finals, scores)
                for method, scores in cold.items()
            },
        }
        cold_curves = [
            mean_curve(runs[f"{method}+cold"], arguments.budget)
            for method in BASELINE_NAMES
        ]
        cold_envelope = [max(values) for values in zip(*cold_curves)]
        prior_curve = mean_curve(runs["PSO+ENS"], arguments.budget)
        stats["anytime_vs_envelope"] = st.mean(
            1.0 if prior_curve[index] >= cold_envelope[index] - 1e-9 else 0.0
            for index in range(arguments.budget)
        )
        stats["best_cold_final"] = max(st.mean(cold[method]) for method in cold)
        pass_points = [
            index + 1
            for index, value in enumerate(prior_curve)
            if value >= stats["best_cold_final"]
        ]
        stats["evals_to_pass_best_cold"] = (
            pass_points[0] if pass_points else arguments.budget + 1
        )
        output["scenarios"][str(scenario_id)] = {
            "name": scenario["name"],
            "use_case": scenario["use_case"],
            "heuristic": hp.evaluate(
                hp.knowledgeable_config(scenario_id), scenario_id, seed=0
            )["score"],
            "runs": runs,
            "stats": stats,
        }
        print(
            f"{scenario_id}: {scenario['name']} "
            f"PSO+ENS={st.mean(ensemble_finals):.2f} "
            f"best cold={stats['best_cold_final']:.2f}",
            flush=True,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260610)
    parser.add_argument("--scenarios", default="1,2,3,4,5,6,7,8")
    parser.add_argument(
        "--llama-results",
        default=str(seed_bank_path(LLAMA_DIR)),
        help=("Llama JSON used as the prior source. The default is the exact "
              "historical ten-configuration seed bank used for Tables 4--6."),
    )
    parser.add_argument(
        "--qwen-results",
        default=str(seed_bank_path(QWEN_DIR)),
        help=("Qwen JSON used as the prior source. The default is the exact "
              "historical ten-configuration seed bank used for Tables 4--6."),
    )
    parser.add_argument(
        "--out",
        default=str(RESULTS_DIR / "reproduced" / "seeded30_results.json"),
    )
    arguments = parser.parse_args()
    arguments.scenarios = [
        int(value) for value in arguments.scenarios.split(",") if value.strip()
    ]
    payload = run(arguments)
    output_path = Path(arguments.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
