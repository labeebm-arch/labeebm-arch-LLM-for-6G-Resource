# SPDX-License-Identifier: MIT
"""Repository path helpers that do not depend on the working directory."""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
TABLES_DIR = REPO_ROOT / "tables"


def result_file(*parts: str) -> Path:
    return RESULTS_DIR.joinpath(*parts)
