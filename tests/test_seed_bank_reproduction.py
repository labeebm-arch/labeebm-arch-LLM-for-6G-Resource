# SPDX-License-Identifier: MIT
"""Deep regression checks for the exact historical seeded-study provenance."""
from __future__ import annotations

import json
import math
from pathlib import Path

from llm6g import hard_problem as hp
from llm6g.paths import RESULTS_DIR
from llm6g.seeded import bayesian_seeded, de_seeded, ga_seeded, pso_seeded
from llm6g.stored_results import (
    LLAMA_DIR,
    QWEN_DIR,
    load_seed_bank,
    sha256_file,
    stored_seed_config,
)
def _close(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
    """Numerically tolerant equality for scalars and (nested) lists of floats."""
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(_close(a, e, rel_tol, abs_tol) for a, e in zip(actual, expected))
    return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def _reference_seeded_results() -> dict:
    path = RESULTS_DIR / "seeded" / "seeded30_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_seed_bank_provenance_hashes_and_protocol():
    provenance_path = RESULTS_DIR / "seed_bank" / "PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["protocol"]["configuration_cycle"] == 10
    assert provenance["protocol"]["optimizer_seed"] == "20260610 + r"
    for model_dir in (LLAMA_DIR, QWEN_DIR):
        item = provenance["files"][model_dir]
        path = RESULTS_DIR.parent / item["path"]
        assert path.is_file()
        assert sha256_file(path) == item["sha256"]
        assert item["repetitions"] == 10
        assert item["mock"] is False

    reference = _reference_seeded_results()
    assert reference["meta"]["llama_seed_bank_sha256"] == provenance["files"][LLAMA_DIR]["sha256"]
    assert reference["meta"]["qwen_seed_bank_sha256"] == provenance["files"][QWEN_DIR]["sha256"]


def test_all_seed_bank_model_configurations_reproduce_stored_scores():
    """Validate all 480 stored zero/few/search configurations and scores."""
    methods = ("LLM-zeroshot", "LLM-fewshot", "LLM-search")
    checked = 0
    for model_dir in (LLAMA_DIR, QWEN_DIR):
        payload = load_seed_bank(model_dir)
        for scenario_id in range(1, 9):
            raw = payload["raw"][str(scenario_id)]
            for method in methods:
                configs = raw[method]["configs"]
                scores = raw[method]["scores"]
                assert len(configs) == len(scores) == 10
                for config, expected in zip(configs, scores):
                    validated = hp.validate(config)
                    actual = hp.evaluate(validated, scenario_id, seed=0)["score"]
                    assert _close(actual, expected)
                    checked += 1
    assert checked == 480


def test_all_reported_pso_seed_histories_reproduce_exactly():
    """Recompute all 960 PSO histories underlying the seed ablation."""
    reference = _reference_seeded_results()
    checked = 0
    for scenario_id in range(1, 9):
        scenario = reference["scenarios"][str(scenario_id)]
        for repetition in range(30):
            llama = stored_seed_config(
                LLAMA_DIR, scenario_id, "LLM-fewshot", repetition
            )
            qwen = stored_seed_config(
                QWEN_DIR, scenario_id, "LLM-fewshot", repetition
            )
            heuristic = hp.knowledgeable_config(scenario_id)
            seed = 20260610 + repetition
            cases = {
                "PSO+L": [llama],
                "PSO+Q": [qwen],
                "PSO+HEUR": [heuristic],
                "PSO+ENS": [qwen, llama, heuristic],
            }
            for tag, configs in cases.items():
                actual = pso_seeded(scenario_id, 60, seed, configs)
                expected = scenario["runs"][tag][repetition]
                assert _close(actual, expected)
                checked += 1
    assert checked == 960


def test_all_reported_ga_and_de_ensemble_histories_reproduce_exactly():
    """Recompute all 480 non-Bayesian host-ablation histories."""
    reference = _reference_seeded_results()
    checked = 0
    for scenario_id in range(1, 9):
        scenario = reference["scenarios"][str(scenario_id)]
        for repetition in range(30):
            llama = stored_seed_config(
                LLAMA_DIR, scenario_id, "LLM-fewshot", repetition
            )
            qwen = stored_seed_config(
                QWEN_DIR, scenario_id, "LLM-fewshot", repetition
            )
            ensemble = [qwen, llama, hp.knowledgeable_config(scenario_id)]
            seed = 20260610 + repetition
            for tag, function in (("GA+ENS", ga_seeded), ("DE+ENS", de_seeded)):
                actual = function(scenario_id, 60, seed, ensemble)
                expected = scenario["runs"][tag][repetition]
                assert _close(actual, expected)
                checked += 1
    assert checked == 480


def test_bayesian_host_first_acquisition_step_reproduces_for_all_scenarios():
    """Check initialization plus one GP acquisition step without a costly full rerun."""
    reference = _reference_seeded_results()
    for scenario_id in range(1, 9):
        llama = stored_seed_config(LLAMA_DIR, scenario_id, "LLM-fewshot", 0)
        qwen = stored_seed_config(QWEN_DIR, scenario_id, "LLM-fewshot", 0)
        ensemble = [qwen, llama, hp.knowledgeable_config(scenario_id)]
        actual = bayesian_seeded(scenario_id, 9, 20260610, ensemble)
        expected = reference["scenarios"][str(scenario_id)]["runs"][
            "Bayesian+ENS"
        ][0][:9]
        assert _close(actual, expected)
