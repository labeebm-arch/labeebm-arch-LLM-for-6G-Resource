# SPDX-License-Identifier: MIT
from __future__ import annotations

import tomllib
from pathlib import Path

import llm6g

from llm6g.paths import REPO_ROOT, RESULTS_DIR
from scripts.run_proxy_study import resolve_config_path


def test_package_version_matches_pyproject():
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    assert llm6g.__version__ == version == "1.0.3"


def test_default_config_resolves_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = resolve_config_path("configs/proxy_study.yaml")
    assert path == REPO_ROOT / "configs" / "proxy_study.yaml"
    assert path.is_file()


def test_reproduced_outputs_are_separate_from_reference_results():
    reproduced = RESULTS_DIR / "reproduced"
    for reference in ("model_campaigns", "seed_bank", "seeded", "ns3", "prior_ns3"):
        assert reproduced != RESULTS_DIR / reference


def test_ns3_rural_mobility_bounds_follow_cell_radius():
    source = (REPO_ROOT / "ns3" / "cttc-nr-llm6g.cc").read_text(encoding="utf-8")
    assert "Rectangle(-cellRadiusM" in source
    assert "Rectangle(-300, 300, -300, 300)" not in source


def test_importing_ns3_runner_does_not_mutate_baseline_backend():
    import importlib

    from llm6g import hard_baselines as hb
    from llm6g import hard_problem as hp
    from scripts import run_ns3_validation as runner

    hb.hp = hp
    importlib.reload(runner)
    assert hb.hp is hp


def test_ns3_capacity_loader_requires_all_requested_scenarios(tmp_path):
    import json

    from scripts.run_ns3_validation import load_capacity_file

    path = tmp_path / "capacity.json"
    path.write_text(json.dumps({"1": 100.0, "2": 80.0}), encoding="utf-8")
    assert load_capacity_file(str(path), [1, 2]) == {1: 100.0, 2: 80.0}

    import pytest

    with pytest.raises(ValueError, match="missing scenarios"):
        load_capacity_file(str(path), [1, 2, 4])


def test_entrypoint_help_is_side_effect_free(tmp_path):
    import os
    import subprocess
    import sys

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + str(REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for script_name in ("reproduce_outputs.py", "run_ns3_oneshot.py"):
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout
        assert "NS3_DIR does not exist" not in completed.stderr
