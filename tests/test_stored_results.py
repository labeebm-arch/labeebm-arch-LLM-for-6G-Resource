# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import statistics as st

import pytest

from llm6g import hard_problem as hp
from llm6g.paths import RESULTS_DIR
from llm6g.stored_results import (
    LLAMA_DIR,
    MODEL_CONFIG_CYCLE,
    QWEN_DIR,
    load_json,
    load_model_campaign,
    load_seed_bank,
    stored_seed_config,
)


def test_model_campaigns_match_table3_protocol():
    for model_dir in [LLAMA_DIR, QWEN_DIR]:
        data = load_model_campaign(model_dir)
        assert data["config"]["repetitions"] == 30
        assert data["config"]["budget"] == 60
        assert set(data["summary"]) >= {str(value) for value in range(1, 9)}


def test_exact_seed_banks_and_modulo_ten_cycle():
    assert MODEL_CONFIG_CYCLE == 10
    for model_dir in [LLAMA_DIR, QWEN_DIR]:
        data = load_seed_bank(model_dir)
        assert data["config"]["repetitions"] == 10
        assert data["config"]["budget"] == 60
        for scenario_id in range(1, 9):
            assert len(data["raw"][str(scenario_id)]["LLM-fewshot"]["configs"]) == 10
        first = stored_seed_config(model_dir, 1, "LLM-fewshot", 0)
        repeated = stored_seed_config(model_dir, 1, "LLM-fewshot", 10)
        assert first == repeated


def test_table3_values_match_final_manuscript():
    llama = load_model_campaign(LLAMA_DIR)
    qwen = load_model_campaign(QWEN_DIR)
    expected = {
        "1": (83.3, 10.8, 42.4, 5.0, 58.7, 60.2, 40.0),
        "2": (86.1, 8.0, 79.6, 6.5, 86.4, 76.7, 6.0),
        "3": (74.8, 34.8, 60.3, 30.5, 60.5, 62.7, 36.5),
        "4": (84.7, 38.8, 88.9, 47.5, 81.2, 76.8, 3.0),
        "5": (74.3, 10.1, 43.2, 5.0, 57.3, 61.1, 61.0),
        "6": (83.6, 14.9, 75.0, 6.0, 67.6, 78.1, 13.5),
        "7": (90.1, 14.8, 76.3, 36.5, 71.6, 73.1, 24.5),
        "8": (85.1, 26.6, 88.0, 56.5, 81.0, 86.6, 47.5),
    }
    for sid, values in expected.items():
        left, right = llama["summary"][sid], qwen["summary"][sid]
        actual = (
            round(left["heuristic_mean"], 1),
            round(left["zeroshot_mean"], 1),
            round(left["fewshot_mean"], 1),
            st.median(left["evals_to_match_fewshot"].values()),
            round(right["zeroshot_mean"], 1),
            round(right["fewshot_mean"], 1),
            st.median(right["evals_to_match_fewshot"].values()),
        )
        assert actual == values


def test_seeded_table_values_match_final_manuscript():
    data = load_json(RESULTS_DIR / "seeded" / "seeded30_results.json")
    assert data["meta"]["seed_cycling"] == "rep r uses stored config r mod 10"
    expected_means = [84.2, 94.8, 78.6, 94.7, 75.2, 91.9, 93.4, 94.1]
    expected_pass = [3, 25, 3, 28, 3, 22, 15, 28]
    for scenario_id, expected_mean, expected_crossing in zip(
        range(1, 9), expected_means, expected_pass
    ):
        scenario = data["scenarios"][str(scenario_id)]
        finals = [history[-1] for history in scenario["runs"]["PSO+ENS"]]
        assert round(st.mean(finals), 1) == expected_mean
        assert scenario["stats"]["evals_to_pass_best_cold"] == expected_crossing
        assert scenario["stats"]["anytime_vs_envelope"] == 1.0


def test_ns3_table_values_match_final_manuscript():
    validation = load_json(RESULTS_DIR / "ns3" / "ns3_validation.json")
    oneshot = load_json(RESULTS_DIR / "ns3" / "ns3_oneshot.json")
    calibration = load_json(RESULTS_DIR / "ns3" / "calibration_summary.json")
    expected = {
        "1": (0.65, 73.1, 73.5, 91.6, 86.1),
        "2": (0.59, 53.9, 52.1, 77.6, 78.3),
        "4": (0.70, 57.8, 74.0, 85.6, 87.2),
    }
    for sid, row in expected.items():
        comparison = validation["comparison"][sid]
        actual = (
            calibration["spearman_rho"][sid],
            round(comparison["heuristic"], 1),
            round(oneshot[sid]["best_oneshot_score"], 1),
            round(comparison["best_cold_final"], 1),
            round(comparison["prior_guided_final"], 1),
        )
        assert actual == row


def test_appendix_c_configurations_are_valid():
    data = load_json(RESULTS_DIR / "model_campaigns" / "best_configurations_table_c.json")
    for config in data["configurations"].values():
        assert hp.validate(config) == config


def test_host_and_seed_ablation_tables_match_final_manuscript():
    data = load_json(RESULTS_DIR / "seeded" / "seeded30_results.json")
    expected_host = {
        "1": [84.2, 83.8, 83.5, 83.3],
        "2": [94.8, 92.7, 90.9, 91.3],
        "3": [78.6, 76.2, 75.7, 75.0],
        "4": [94.7, 92.8, 92.3, 93.1],
        "5": [75.2, 74.8, 74.3, 74.3],
        "6": [91.9, 90.0, 88.0, 88.7],
        "7": [93.4, 92.4, 91.2, 92.5],
        "8": [94.1, 92.2, 91.4, 92.4],
    }
    expected_seed = {
        "1": [69.5, 77.7, 79.1, 84.2, 84.2],
        "2": [91.2, 93.0, 92.1, 94.5, 94.8],
        "3": [73.7, 76.6, 77.7, 78.4, 78.6],
        "4": [92.7, 94.6, 93.3, 94.8, 94.7],
        "5": [64.1, 70.3, 73.8, 75.1, 75.2],
        "6": [87.4, 88.3, 88.0, 92.0, 91.9],
        "7": [90.6, 92.2, 92.0, 93.4, 93.4],
        "8": [91.1, 93.5, 93.6, 94.2, 94.1],
    }
    for sid, scenario in data["scenarios"].items():
        runs = scenario["runs"]
        host = [
            round(st.mean(history[-1] for history in runs[tag]), 1)
            for tag in ["PSO+ENS", "GA+ENS", "DE+ENS", "Bayesian+ENS"]
        ]
        seed = [
            round(scenario["stats"]["best_cold_final"], 1),
            *[
                round(st.mean(history[-1] for history in runs[tag]), 1)
                for tag in ["PSO+L", "PSO+Q", "PSO+HEUR", "PSO+ENS"]
            ],
        ]
        assert host == expected_host[sid]
        assert seed == expected_seed[sid]


def test_figure1_historical_curves_match_manuscript_annotations():
    data = load_json(RESULTS_DIR / "prior_ns3" / "figure1_histories.json")
    expected = {
        "1": (97.9, 98),
        "2": (91.5, 96),
    }
    for sid, (score, percentage) in expected.items():
        scenario = data["scenarios"][sid]
        curve = scenario["methods"]["PSO"]
        assert len(curve) == 60
        assert round(curve[7], 1) == score
        assert round(100 * curve[7] / curve[-1]) == percentage
