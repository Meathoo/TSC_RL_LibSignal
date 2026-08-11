#!/usr/bin/env python3
"""Continue completed MAPPO-IRU runs from an episode-boundary checkpoint.

The source run is never modified.  Its checkpoint is copied into a new output
prefix, then ``run.py`` is invoked with matching architecture/ID settings and a
resume episode.  This restores model, learned agent embedding, and optimizer
state.  RNG and simulator state are intentionally not restored, so this is a
practical warm resume rather than a bitwise-identical continuation.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_PATTERNS = (
    "seed*_mlp_learned_Equal2NativeMAPPO_ep100",
    "seed*_actor_iru1_learned_ep100",
    "seed*_actor_iru5_learned_ep100",
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Resume MAPPO-IRU runs into new output prefixes."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "data/output_data/tsc/cityflow_mappo_iru/cityflow16x3",
        help="Directory containing the source run directories.",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=[],
        help="Explicit source run directory; repeat for multiple runs.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob under --root. Defaults to MLP, actor IRU n1, and actor IRU n5.",
    )
    parser.add_argument("--from-episode", type=int, default=100)
    parser.add_argument("--to-episode", type=int, default=250)
    parser.add_argument(
        "--suffix",
        default=None,
        help="Target prefix suffix; defaults to _resume<to-episode>.",
    )
    parser.add_argument(
        "--ngpu",
        default=None,
        help="GPU override. By default reuse the source command value.",
    )
    parser.add_argument(
        "--profile-performance",
        type=str2bool,
        default=None,
        help="Performance profiling override; by default reuse the source setting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoints and print commands without copying or running.",
    )
    return parser.parse_args()


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_sources(args):
    sources = [path.resolve() for path in args.source]
    patterns = args.pattern or DEFAULT_PATTERNS
    if not args.source:
        for pattern in patterns:
            sources.extend(sorted(args.root.resolve().glob(pattern)))
    unique = []
    seen = set()
    for source in sources:
        if source not in seen:
            seen.add(source)
            unique.append(source)
    if not unique:
        raise FileNotFoundError("No source runs matched the requested paths/patterns")
    return unique


def bool_arg(value):
    return "true" if bool(value) else "false"


def require_choice(value, allowed, label):
    normalized = str(value).lower()
    if normalized not in allowed:
        raise ValueError(f"Unsupported {label} in source hyperparameters: {value!r}")
    return normalized


def prepare_run(source, args, repo_root):
    hyper_path = source / "hyperparameters.json"
    if not hyper_path.is_file():
        raise FileNotFoundError(f"Missing source hyperparameters: {hyper_path}")
    hyper = json.loads(hyper_path.read_text(encoding="utf-8"))
    command = hyper.get("command", {})
    model = hyper.get("model", {})

    source_checkpoints = sorted((source / "model").glob(f"{args.from_episode}_*.pt"))
    if not source_checkpoints:
        raise FileNotFoundError(
            f"No episode-{args.from_episode} checkpoint found under {source / 'model'}"
        )
    if args.to_episode <= args.from_episode:
        raise ValueError("--to-episode must be greater than --from-episode")

    source_prefix = str(command.get("prefix") or source.name)
    suffix = args.suffix or f"_resume{args.to_episode}"
    target_prefix = source_prefix + suffix
    task = str(command.get("task", "tsc"))
    agent = str(command.get("agent", "mappo_iru"))
    world = str(command.get("world", "cityflow"))
    network = str(command.get("network", source.parent.name))
    seed = command.get("seed")
    if seed is None:
        raise ValueError(f"Source run has no seed: {source}")

    actor_arch = require_choice(model.get("native_actor_arch", "iru"), {"mlp", "iru"}, "actor architecture")
    value_arch = require_choice(model.get("native_value_arch", "iru"), {"mlp", "iru"}, "value architecture")
    id_mode = require_choice(model.get("native_agent_id_mode", "one_hot"), {"one_hot", "learned"}, "agent ID mode")
    use_agent_id = bool(model.get("native_use_agent_id", False))
    actor_steps = int(model.get("iru_actor_steps", model.get("iru_steps", 1)))
    value_steps = int(model.get("iru_value_steps", model.get("iru_steps", 1)))
    hidden_dim = int(model.get("iru_hidden_dim", 64))
    num_blocks = int(model.get("iru_num_blocks", 1))
    profile = (
        bool(model.get("profile_performance", False))
        if args.profile_performance is None
        else args.profile_performance
    )
    ngpu = str(command.get("ngpu", "0") if args.ngpu is None else args.ngpu)

    canonical_parent = (
        repo_root
        / "data/output_data"
        / task
        / f"{world}_{agent}"
        / network
    )
    canonical_source = canonical_parent / source_prefix
    if source.resolve() != canonical_source.resolve():
        raise ValueError(
            "Source path does not match its saved command metadata: "
            f"path={source.resolve()}, expected={canonical_source.resolve()}"
        )
    target = canonical_parent / target_prefix
    target_checkpoints = target / "model"
    if target.exists():
        raise FileExistsError(
            f"Target already exists: {target}. Choose another --suffix or remove it explicitly."
        )

    run_command = [
        sys.executable,
        "-u",
        "run.py",
        "--task",
        task,
        "--agent",
        agent,
        "--world",
        world,
        "--network",
        network,
        "--prefix",
        target_prefix,
        "--seed",
        str(seed),
        "--ngpu",
        ngpu,
        "--native_actor_arch",
        actor_arch,
        "--native_value_arch",
        value_arch,
        "--iru_actor_steps",
        str(actor_steps),
        "--iru_value_steps",
        str(value_steps),
        "--iru_hidden_dim",
        str(hidden_dim),
        "--iru_num_blocks",
        str(num_blocks),
        "--native_use_agent_id",
        bool_arg(use_agent_id),
        "--native_agent_id_mode",
        id_mode,
        "--profile_performance",
        bool_arg(profile),
        "--episodes",
        str(args.to_episode),
        "--resume_episode",
        str(args.from_episode),
        "--config_snapshot",
        str(hyper_path.resolve()),
    ]
    return target, target_checkpoints, source_checkpoints, run_command


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sources = collect_sources(args)
    print(
        f"Found {len(sources)} source run(s); continuing episode "
        f"{args.from_episode} -> {args.to_episode}.",
        flush=True,
    )

    for index, source in enumerate(sources, start=1):
        target, target_model_dir, checkpoints, command = prepare_run(
            source, args, repo_root
        )
        print(f"[{index}/{len(sources)}] source: {source}", flush=True)
        print(f"[{index}/{len(sources)}] target: {target}", flush=True)
        print(
            f"[{index}/{len(sources)}] command: {subprocess.list2cmdline(command)}",
            flush=True,
        )
        if args.dry_run:
            continue

        target_model_dir.mkdir(parents=True, exist_ok=False)
        copied = []
        for checkpoint in checkpoints:
            destination = target_model_dir / checkpoint.name
            shutil.copy2(checkpoint, destination)
            copied.append(
                {
                    "source": str(checkpoint),
                    "destination": str(destination),
                    "sha256": checkpoint_digest(destination),
                }
            )
        metadata = {
            "source_run": str(source),
            "target_run": str(target),
            "from_episode": args.from_episode,
            "to_episode": args.to_episode,
            "checkpoints": copied,
            "note": "Model/optimizer resumed; RNG and simulator state were not restored.",
        }
        (target / "resume_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        subprocess.run(command, cwd=repo_root, check=True)

    print("Resume jobs completed." if not args.dry_run else "Dry run completed.")


if __name__ == "__main__":
    main()
