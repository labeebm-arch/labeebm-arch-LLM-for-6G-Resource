#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stdlib-only check that must run before editable installation."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIR_NAMES = {"__pycache__", ".pytest_cache", "build", "dist", ".mypy_cache"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}


def main() -> None:
    forbidden: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        relative = path.relative_to(REPO_ROOT)
        if path.is_dir() and (
            path.name in FORBIDDEN_DIR_NAMES or path.name.endswith(".egg-info")
        ):
            forbidden.append(relative)
        elif path.is_file() and path.suffix in FORBIDDEN_SUFFIXES:
            forbidden.append(relative)
    if forbidden:
        print("The release tree contains generated build artifacts:", file=sys.stderr)
        for path in sorted(set(forbidden)):
            print(f"- {path}", file=sys.stderr)
        raise SystemExit(1)
    print("Release tree is clean before installation.")


if __name__ == "__main__":
    main()
