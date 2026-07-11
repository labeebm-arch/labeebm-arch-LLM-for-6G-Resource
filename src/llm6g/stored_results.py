# SPDX-License-Identifier: MIT
"""Load immutable campaign results and the exact historical seeded-study priors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from . import hard_problem as hp
from .paths import RESULTS_DIR

LLAMA_DIR = "llama3_8b"
QWEN_DIR = "qwen2.5_7b-instruct-q5_K_M"
MODEL_CONFIG_CYCLE = 10
MODEL_CAMPAIGNS_DIR = RESULTS_DIR / "model_campaigns"
SEED_BANK_DIR = RESULTS_DIR / "seed_bank"
SEEDED_RESULTS_DIR = RESULTS_DIR / "seeded"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_campaign_path(model_dir: str) -> Path:
    """Return the 30-run campaign file used for Table 3."""
    return MODEL_CAMPAIGNS_DIR / model_dir / "hard_comparison_results.json"


def seed_bank_path(model_dir: str) -> Path:
    """Return the exact 10-configuration bank used for Tables 4--6."""
    return SEED_BANK_DIR / model_dir / "hard_comparison_results.json"


def load_model_campaign(model_dir: str) -> Dict[str, Any]:
    path = model_campaign_path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing stored 30-run model campaign: {path}")
    return load_json(path)


def load_seed_bank(model_dir: str) -> Dict[str, Any]:
    path = seed_bank_path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing exact historical seed bank: {path}")
    return load_json(path)


def stored_seed_config(
    model_dir: str, scenario_id: int, method: str, repetition: int
) -> Dict:
    """Return the paper's prior for a seeded repetition.

    The final manuscript cycles over ten historical model configurations.
    Repetition ``r`` uses configuration index ``r % 10`` while the optimizer
    seed changes on every repetition. These ten-run files are distinct from the
    30-run model campaigns used to report Table 3.
    """
    data = load_seed_bank(model_dir)
    configs = data["raw"][str(scenario_id)][method]["configs"]
    if len(configs) != MODEL_CONFIG_CYCLE:
        raise ValueError(
            f"{model_dir}, scenario {scenario_id}, {method} contains "
            f"{len(configs)} configurations; exactly {MODEL_CONFIG_CYCLE} are required."
        )
    return hp.validate(configs[repetition % MODEL_CONFIG_CYCLE])


def best_campaign_config(model_dir: str, scenario_id: int, method: str) -> Dict:
    data = load_model_campaign(model_dir)
    config = data["summary"][str(scenario_id)]["rows"][method]["best_config"]
    return hp.validate(config)
