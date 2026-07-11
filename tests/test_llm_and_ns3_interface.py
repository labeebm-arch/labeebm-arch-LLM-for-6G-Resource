# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from llm6g import hard_llm as hl
from llm6g import hard_problem as hp
from llm6g import ns3_problem as n3


def test_malformed_reply_is_not_silently_randomized(monkeypatch):
    monkeypatch.setattr(hl, "query_ollama", lambda *args, **kwargs: "not JSON")
    with pytest.raises(hl.LLMParseError):
        hl.llm_zeroshot(1, "http://localhost:11434", "model", mock=False)
    with pytest.raises(hl.LLMParseError):
        hl.llm_fewshot(1, hp.SCENARIOS[1:4], "http://localhost:11434", "model", mock=False)


def test_ns3_command_maps_all_twelve_parameters():
    params = hp.knowledgeable_config(1)
    args = n3._build_args(params, hp.SCENARIO_BY_ID[1], 1.0, 1)
    joined = " ".join(args)
    expected_names = {
        "scheduler", "numUes", "txPower", "bandwidthMHz", "packetSize",
        "trafficRateMbps", "numerology", "mimoLayers", "harqProcesses",
        "targetBler", "bandSplit", "txAntennas",
    }
    for name in expected_names:
        assert f"--{name}=" in joined


def test_ns3_capacity_file_is_repository_relative():
    assert n3._CAP_FILE.endswith("results/ns3/ns3_capacity.json")


def test_new_proxy_runs_do_not_target_reference_results():
    from scripts.run_proxy_study import default_output_path

    config = {
        "ollama": {"model": "llama3:8b"},
        "mock": False,
    }
    path = default_output_path(config)
    assert "results/reproduced" in str(path).replace("\\", "/")
    assert path.name == "hard_comparison_results.json"


def test_partial_json_reply_is_rejected():
    complete = hp.knowledgeable_config(1)
    incomplete = dict(complete)
    incomplete.pop("targetBler")
    import json

    assert hl.parse_configs(json.dumps(incomplete)) == []
    assert hl.parse_configs(json.dumps(complete)) == [hp.validate(complete)]


def test_ns3_capacity_persistence_is_opt_in():
    import inspect

    assert inspect.signature(n3.set_ns3_capacity).parameters["persist"].default is False
