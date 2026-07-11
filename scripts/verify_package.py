#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run structural and manuscript-consistency checks for the public package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import yaml

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "RELEASE_NOTES.md",
    "AUDIT_REPORT.md",
    "LICENSE",
    "LICENSES/MIT.txt",
    "LICENSES/GPL-2.0-only.txt",
    "LICENSES/CC-BY-4.0.txt",
    "CITATION.cff",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
    "MANIFEST.sha256",
    ".github/workflows/tests.yml",
    "configs/proxy_study.yaml",
    "docs/MANUSCRIPT_ALIGNMENT.md",
    "results/README.md",
    "scripts/check_release_tree.py",
    "results/model_campaigns/llama3_8b/hard_comparison_results.json",
    "results/model_campaigns/qwen2.5_7b-instruct-q5_K_M/hard_comparison_results.json",
    "results/seed_bank/llama3_8b/hard_comparison_results.json",
    "results/seed_bank/qwen2.5_7b-instruct-q5_K_M/hard_comparison_results.json",
    "results/seed_bank/PROVENANCE.json",
    "results/seeded/seeded30_results.json",
    "results/ns3/ns3_validation.json",
    "results/ns3/ns3_oneshot.json",
    "results/prior_ns3/figure1_histories.json",
    "ns3/cttc-nr-llm6g.cc",
]

GENERATED_PREFIXES = (
    "figures/generated/",
    "results/reproduced/",
    "tables/",
)


def is_manifest_candidate(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT).as_posix()
    if relative == "MANIFEST.sha256":
        return False
    if relative.startswith(GENERATED_PREFIXES):
        return False
    if any(part in {"__pycache__", ".pytest_cache", "build", "dist", ".git"} for part in path.parts):
        return False
    if path.suffix in {".pyc", ".pyo"} or any(part.endswith(".egg-info") for part in path.parts):
        return False
    return True


def verify_manifest() -> None:
    manifest = REPO_ROOT / "MANIFEST.sha256"
    failures: list[str] = []
    listed: set[str] = set()
    checked = 0
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(None, 1)
        relative = relative.removeprefix("*").removeprefix("./")
        if relative in listed:
            failures.append(f"duplicate manifest entry: {relative}")
            continue
        listed.add(relative)
        path = REPO_ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checked += 1
        if actual != expected:
            failures.append(f"checksum mismatch: {relative}")

    expected_files = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.rglob("*")
        if path.is_file() and is_manifest_candidate(path)
    }
    for relative in sorted(expected_files - listed):
        failures.append(f"immutable file omitted from manifest: {relative}")
    for relative in sorted(listed - expected_files):
        failures.append(f"unexpected manifest entry: {relative}")

    if failures:
        raise SystemExit("Manifest verification failed:\n- " + "\n- ".join(failures))
    print(f"Validated {checked} immutable-file checksums with complete coverage.")


def local_environment() -> dict[str, str]:
    environment = dict(os.environ)
    local_src = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = local_src + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Avoid severe BLAS oversubscription during deterministic GP regression tests.
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    return environment


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        env=local_environment(),
    )


def compile_python_sources() -> None:
    sources = sorted(
        path
        for folder in ("src", "scripts", "tests")
        for path in (REPO_ROOT / folder).rglob("*.py")
    )
    with tempfile.TemporaryDirectory(prefix="llm6g-pyc-") as directory:
        cache = Path(directory)
        for index, source in enumerate(sources):
            py_compile.compile(
                str(source),
                cfile=str(cache / f"{index}.pyc"),
                doraise=True,
            )
    print(f"Compiled {len(sources)} Python files without writing into the repository.")


def verify_version_consistency() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        project_version = tomllib.load(handle)["project"]["version"]
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import llm6g

    imported_root = Path(llm6g.__file__).resolve().parents[2]
    if imported_root != REPO_ROOT.resolve():
        raise SystemExit(f"Imported llm6g from the wrong repository: {llm6g.__file__}")
    if llm6g.__version__ != project_version:
        raise SystemExit(
            f"Version mismatch: pyproject={project_version}, llm6g={llm6g.__version__}"
        )
    citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    if str(citation.get("version")) != project_version:
        raise SystemExit(
            f"Version mismatch: pyproject={project_version}, CITATION.cff={citation.get('version')}"
        )
    print(f"Validated package and citation version {project_version}.")


def verify_no_build_artifacts() -> None:
    forbidden = []
    for path in REPO_ROOT.rglob("*"):
        relative = path.relative_to(REPO_ROOT)
        if path.is_dir() and (
            path.name in {"__pycache__", ".pytest_cache", "build", "dist"}
        ):
            forbidden.append(relative)
        elif path.is_file() and path.suffix in {".pyc", ".pyo"}:
            forbidden.append(relative)
    if forbidden:
        raise SystemExit(
            "Generated build artifacts are present:\n- "
            + "\n- ".join(map(str, sorted(set(forbidden))))
        )
    print("Validated clean release tree.")


def verify_licensing() -> None:
    ns3_source = (REPO_ROOT / "ns3" / "cttc-nr-llm6g.cc").read_text(encoding="utf-8")
    if not any(
        "SPDX-License-Identifier: GPL-2.0-only" in line
        for line in ns3_source.splitlines()[:10]
    ):
        raise SystemExit("The ns-3 integration file lacks its GPL-2.0-only SPDX header.")
    for folder in ("src", "scripts", "tests"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            lines = path.read_text(encoding="utf-8").splitlines()[:5]
            if not any("SPDX-License-Identifier: MIT" in line for line in lines):
                raise SystemExit(f"Missing MIT SPDX header: {path.relative_to(REPO_ROOT)}")
    print("Validated mixed-license files and SPDX headers.")


def verify_structured_files() -> None:
    json_files = sorted((REPO_ROOT / "results").rglob("*.json"))
    for path in json_files:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)

    yaml_files = [
        REPO_ROOT / "configs" / "proxy_study.yaml",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / ".github" / "workflows" / "tests.yml",
    ]
    for path in yaml_files:
        yaml.safe_load(path.read_text(encoding="utf-8"))

    csv_files = sorted((REPO_ROOT / "tables").glob("*.csv"))
    for path in csv_files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2 or not rows[0]:
            raise SystemExit(f"CSV is empty or malformed: {path.relative_to(REPO_ROOT)}")
    print(
        f"Validated {len(json_files)} JSON, {len(yaml_files)} YAML/CFF, "
        f"and {len(csv_files)} CSV files."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-regeneration",
        action="store_true",
        help="Do not rebuild tables and generated figures.",
    )
    arguments = parser.parse_args()

    missing = [name for name in REQUIRED_FILES if not (REPO_ROOT / name).exists()]
    if missing:
        raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))

    verify_no_build_artifacts()
    verify_version_consistency()
    verify_licensing()
    verify_manifest()
    verify_structured_files()
    compile_python_sources()

    run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    if not arguments.skip_regeneration:
        run([sys.executable, "scripts/reproduce_outputs.py"])
    print("Package verification passed.")


if __name__ == "__main__":
    main()
