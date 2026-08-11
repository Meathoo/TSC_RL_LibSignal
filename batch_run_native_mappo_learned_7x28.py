#!/usr/bin/env python3
"""Complete native MAPPO learned64 seeds 1 and 2 on CityFlow 7x28.

The seed0 hyperparameter snapshot is used as the effective configuration.  Each
run is first produced under the standard ``cityflow_native_mappo_learned``
output path, then organized beside the existing seed0 reference under
``cityflow_native_mappo/.../queue+phase+learned64(id)``.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
REFERENCE_RUN = (
    REPO_ROOT
    / "data/output_data/tsc/cityflow_native_mappo/cityflow7x28"
    / "queue+phase+learned64(id)/seed0_learned64"
)
CONFIG_SNAPSHOT = REFERENCE_RUN / "hyperparameters.json"
STANDARD_ROOT = (
    REPO_ROOT
    / "data/output_data/tsc/cityflow_native_mappo_learned/cityflow7x28"
)
FINAL_ROOT = REFERENCE_RUN.parent
SEEDS = (1, 2)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_against_reference(run_dir, seed):
    candidate_path = run_dir / "hyperparameters.json"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Missing completed hyperparameters: {candidate_path}")

    reference = load_json(CONFIG_SNAPSHOT)
    candidate = load_json(candidate_path)
    errors = []
    for section in ("model", "trainer", "logger"):
        if candidate.get(section) != reference.get(section):
            errors.append(f"{section} differs from seed0")

    expected_command = {
        "task": "tsc",
        "agent": "native_mappo_learned",
        "world": "cityflow",
        "network": "cityflow7x28",
        "prefix": f"seed{seed}_learned64",
        "seed": seed,
        "ngpu": "0",
    }
    command = candidate.get("command", {})
    for key, expected in expected_command.items():
        if command.get(key) != expected:
            errors.append(
                f"command.{key}: expected={expected!r}, actual={command.get(key)!r}"
            )

    reference_world = reference.get("world", {})
    candidate_world = candidate.get("world", {})
    for key in (
        "network",
        "interval",
        "dir",
        "roadnetFile",
        "flowFile",
        "rlTrafficLight",
    ):
        if candidate_world.get(key) != reference_world.get(key):
            errors.append(
                f"world.{key}: seed0={reference_world.get(key)!r}, "
                f"candidate={candidate_world.get(key)!r}"
            )
    if candidate_world.get("seed") != seed:
        errors.append(
            f"world.seed: expected={seed}, actual={candidate_world.get('seed')!r}"
        )

    if errors:
        raise ValueError(
            f"Hyperparameter validation failed for seed{seed}:\n- "
            + "\n- ".join(errors)
        )


def completed(run_dir):
    hyperparameters = run_dir / "hyperparameters.json"
    model = run_dir / "model/250_0.pt"
    dtl_logs = list((run_dir / "logger").glob("*_DTL.log"))
    return hyperparameters.is_file() and model.is_file() and bool(dtl_logs)


def build_command(seed):
    prefix = f"seed{seed}_learned64"
    return [
        sys.executable,
        "-u",
        "run.py",
        "--task",
        "tsc",
        "--agent",
        "native_mappo_learned",
        "--world",
        "cityflow",
        "--network",
        "cityflow7x28",
        "--prefix",
        prefix,
        "--seed",
        str(seed),
        "--ngpu",
        "0",
        "--config_snapshot",
        str(CONFIG_SNAPSHOT),
    ]


def run_seed(seed, dry_run=False):
    prefix = f"seed{seed}_learned64"
    standard_run = STANDARD_ROOT / prefix
    final_run = FINAL_ROOT / prefix
    command = build_command(seed)

    if dry_run:
        print("[dry-run] " + subprocess.list2cmdline(command), flush=True)
        print(f"[dry-run] organize output: {standard_run} -> {final_run}", flush=True)
        return

    if completed(final_run):
        validate_against_reference(final_run, seed)
        print(f"[skip] seed{seed} already completed and validated: {final_run}", flush=True)
        return
    if final_run.exists():
        raise FileExistsError(f"Incomplete destination already exists: {final_run}")
    if standard_run.exists():
        raise FileExistsError(
            f"Temporary standard output already exists; inspect it before retrying: "
            f"{standard_run}"
        )

    print("[run] " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    validate_against_reference(standard_run, seed)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.move(str(standard_run), str(final_run))
    validate_against_reference(final_run, seed)
    print(f"[done] seed{seed}: {final_run}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not CONFIG_SNAPSHOT.is_file():
        raise FileNotFoundError(f"Missing seed0 reference: {CONFIG_SNAPSHOT}")
    for seed in SEEDS:
        run_seed(seed, dry_run=args.dry_run)
    print(
        "Dry run completed."
        if args.dry_run
        else "All native MAPPO learned64 7x28 seeds completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
