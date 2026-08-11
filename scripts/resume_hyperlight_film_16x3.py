#!/usr/bin/env python3
"""Continue the three FiLM (hyperlight_film_both_mlp) 16x3 runs from ep250 to ep350.

Mirrors scripts/resume_mappo_iru.py: the source run is never modified. Its
ep250 checkpoint is copied into a new output prefix, then run.py is invoked
with --config_snapshot pointing at the source hyperparameters.json (so every
hyper_* override, e.g. hyper_film_scale, is restored verbatim) plus
--resume_episode/--episodes to continue training. RNG and simulator state are
not restored, so this is a practical warm resume, not a bitwise continuation.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FROM_EPISODE = 250
TO_EPISODE = 350
SOURCES = [
    REPO_ROOT
    / "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3"
    / f"hyperlight_film_both_mlp_seed{seed}"
    for seed in (0, 1, 2)
]


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_run(source):
    hyper_path = source / "hyperparameters.json"
    hyper = json.loads(hyper_path.read_text(encoding="utf-8"))
    command = hyper.get("command", {})

    checkpoint = source / "model" / f"{FROM_EPISODE}_0.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing episode-{FROM_EPISODE} checkpoint: {checkpoint}")

    source_prefix = str(command.get("prefix") or source.name)
    target_prefix = f"{source_prefix}_resume{TO_EPISODE}"
    target = source.parent / target_prefix
    if target.exists():
        raise FileExistsError(f"Target already exists: {target}")

    run_command = [
        sys.executable,
        "-u",
        "run.py",
        "--task", str(command.get("task", "tsc")),
        "--agent", str(command.get("agent", "hyperlight_mappo")),
        "--world", str(command.get("world", "cityflow")),
        "--network", str(command.get("network", "cityflow16x3")),
        "--prefix", target_prefix,
        "--seed", str(command.get("seed")),
        "--ngpu", str(command.get("ngpu", "0")),
        "--episodes", str(TO_EPISODE),
        "--resume_episode", str(FROM_EPISODE),
        "--config_snapshot", str(hyper_path.resolve()),
    ]
    return target, checkpoint, run_command


def main():
    for index, source in enumerate(SOURCES, start=1):
        target, checkpoint, command = prepare_run(source)
        print(f"[{index}/{len(SOURCES)}] source: {source}", flush=True)
        print(f"[{index}/{len(SOURCES)}] target: {target}", flush=True)
        print(f"[{index}/{len(SOURCES)}] command: {subprocess.list2cmdline(command)}", flush=True)

        model_dir = target / "model"
        model_dir.mkdir(parents=True, exist_ok=False)
        destination = model_dir / checkpoint.name
        shutil.copy2(checkpoint, destination)
        metadata = {
            "source_run": str(source),
            "target_run": str(target),
            "from_episode": FROM_EPISODE,
            "to_episode": TO_EPISODE,
            "checkpoint": {
                "source": str(checkpoint),
                "destination": str(destination),
                "sha256": checkpoint_digest(destination),
            },
            "note": "Model/optimizer resumed; RNG and simulator state were not restored.",
        }
        (target / "resume_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        subprocess.run(command, cwd=REPO_ROOT, check=True)
        print(f"[{index}/{len(SOURCES)}] done: {target}", flush=True)

    print("All resume jobs completed.")


if __name__ == "__main__":
    main()
