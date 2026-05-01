#!/usr/bin/env python3
"""
Evaluate a saved HyperLight checkpoint with explicit config/dataset loading.

This entrypoint is intentionally separate from run.py so evaluation can load a
checkpoint from one output prefix while writing logs/results to another prefix.
It still uses the project's normal Registry, config, dataset, world, trainer,
agent, and metrics pipeline.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_seeds(raw: Iterable[str]) -> List[int]:
    seeds: List[int] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            if part:
                seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate HyperLight best/final checkpoints on one or more seeds."
    )

    parser.add_argument("--agent", default="hyperlight", help="Agent config/model name.")
    parser.add_argument("--task", default="tsc", help="Task name.")
    parser.add_argument("--world", default="cityflow", choices=["cityflow", "sumo"])
    parser.add_argument("--network", default="cityflow7x28", help="Target simulator config name.")
    parser.add_argument("--dataset", default="onfly", help="Dataset backend registered in dataset/.")
    parser.add_argument("--prefix", default="hyperlight_cf2_eval", help="Evaluation output prefix.")

    parser.add_argument(
        "--source-prefix",
        default="hyperlight_cf2",
        help="Training output prefix that contains the checkpoint model/ directory.",
    )
    parser.add_argument(
        "--source-network",
        default=None,
        help="Network of the source checkpoint. Defaults to --network.",
    )
    parser.add_argument(
        "--source-agent",
        default=None,
        help="Agent of the source checkpoint. Defaults to --agent.",
    )
    parser.add_argument(
        "--source-world",
        default=None,
        help="World of the source checkpoint. Defaults to --world.",
    )
    parser.add_argument(
        "--checkpoint-output-dir",
        default=None,
        help="Optional direct path to the source run directory containing model/.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint tag, e.g. best, 200, 140. Loads <checkpoint>_<rank>.pt.",
    )
    parser.add_argument(
        "--no-source-hyperparams",
        action="store_true",
        help="Do not reuse source hyperparameters.json model/trainer settings.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0"],
        help="Evaluation seeds. Accepts space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--single-prefix",
        action="store_true",
        help="Write all seeds under exactly --prefix instead of appending _seed<seed>.",
    )
    parser.add_argument("--thread-num", type=int, default=4, help="CityFlow thread count.")
    parser.add_argument("--ngpu", default="0", help="GPU id visible to PyTorch. Use -1 for CPU.")
    parser.add_argument("--cpu", action="store_true", help="Force HyperLight to run on CPU.")
    parser.add_argument("--debug", action="store_true", help="Use DEBUG logging.")
    parser.add_argument("--interface", default="libsumo", choices=["libsumo", "traci"])
    parser.add_argument("--delay-type", default="apx", choices=["apx", "real"])

    parser.add_argument("--test-steps", type=int, default=None, help="Override trainer.test_steps.")
    parser.add_argument(
        "--action-interval",
        type=int,
        default=None,
        help="Override trainer.action_interval.",
    )
    parser.add_argument("--roadnet-file", default=None, help="Override world.roadnetFile.")
    parser.add_argument("--flow-file", default=None, help="Override world.flowFile.")
    parser.add_argument("--world-dir", default=None, help="Override world.dir.")
    parser.add_argument(
        "--save-replay",
        action="store_true",
        help="Enable simulator replay output for evaluation runs.",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Optional CSV summary path. Defaults under data/output_data/.../<network>/.",
    )
    return parser


def resolve_source_output_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_output_dir:
        direct_path = Path(args.checkpoint_output_dir).expanduser().resolve()
        if direct_path.name == "model":
            return direct_path.parent
        return direct_path

    source_world = args.source_world or args.world
    source_agent = args.source_agent or args.agent
    source_network = args.source_network or args.network
    return (
        REPO_ROOT
        / "data"
        / "output_data"
        / args.task
        / f"{source_world}_{source_agent}"
        / source_network
        / args.source_prefix
    ).resolve()


def load_source_hyperparameters(source_output_dir: Path) -> Optional[Dict[str, Any]]:
    hyperparam_path = source_output_dir / "hyperparameters.json"
    if not hyperparam_path.exists():
        return None
    with hyperparam_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_run_args(args: argparse.Namespace, seed: int, prefix: str) -> SimpleNamespace:
    return SimpleNamespace(
        thread_num=args.thread_num,
        ngpu=args.ngpu,
        prefix=prefix,
        seed=seed,
        debug=args.debug,
        interface=args.interface,
        delay_type=args.delay_type,
        task=args.task,
        agent=args.agent,
        world=args.world,
        network=args.network,
        dataset=args.dataset,
    )


def apply_eval_overrides(
    config: Dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    source_hyperparams: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    config = deepcopy(config)

    if source_hyperparams is not None and not args.no_source_hyperparams:
        if "model" in source_hyperparams:
            config["model"] = deepcopy(source_hyperparams["model"])
        if "trainer" in source_hyperparams:
            config["trainer"].update(deepcopy(source_hyperparams["trainer"]))

    config["command"]["seed"] = seed
    config["command"]["ngpu"] = args.ngpu
    config["command"]["thread_num"] = args.thread_num
    config["command"]["dataset"] = args.dataset

    config["model"]["name"] = args.agent
    config["model"]["train_model"] = False
    config["model"]["test_model"] = True
    config["model"]["load_model"] = True
    if args.cpu:
        config["model"]["use_cuda"] = False

    config["trainer"]["test_when_train"] = False
    config["trainer"]["load_best_for_test"] = False
    if args.test_steps is not None:
        config["trainer"]["test_steps"] = int(args.test_steps)
    if args.action_interval is not None:
        config["trainer"]["action_interval"] = int(args.action_interval)

    config["world"]["seed"] = int(seed)
    config["world"]["saveReplay"] = bool(args.save_replay)
    if args.roadnet_file is not None:
        config["world"]["roadnetFile"] = args.roadnet_file
    if args.flow_file is not None:
        config["world"]["flowFile"] = args.flow_file
    if args.world_dir is not None:
        config["world"]["dir"] = args.world_dir

    return config


def reset_logging_handlers() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        try:
            handler.flush()
            handler.close()
        finally:
            root.removeHandler(handler)


def configure_registry(config: Dict[str, Any]) -> None:
    from common import interface
    from common.registry import Registry

    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config)

    if config["model"].get("graphic", False):
        world_param = Registry.mapping["world_mapping"]["setting"].param
        if config["command"]["world"] in ["cityflow", "sumo"]:
            roadnet_path = os.path.join(world_param["dir"], world_param["roadnetFile"])
        else:
            roadnet_path = world_param["road_file_addr"]
        interface.Graph_World_Interface(roadnet_path)

    interface.Logger_path_Interface(config)
    os.makedirs(Registry.mapping["logger_mapping"]["path"].path, exist_ok=True)
    interface.Trainer_param_Interface(config)
    interface.ModelAgent_param_Interface(config)


def setup_project_modules() -> None:
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Import side effects register task/trainer/agent/world/dataset classes.
    import agent  # noqa: F401
    import dataset  # noqa: F401
    import task  # noqa: F401
    import trainer  # noqa: F401
    import world  # noqa: F401


def load_agents_from_source(trainer_obj: Any, checkpoint: str, source_output_dir: Path) -> None:
    from common.registry import Registry

    model_dir = source_output_dir / "model"
    missing = []
    for ag in trainer_obj.agents:
        candidate = model_dir / f"{checkpoint}_{ag.rank}.pt"
        if not candidate.exists():
            missing.append(str(candidate))
    if missing:
        raise FileNotFoundError("Missing checkpoint file(s): " + ", ".join(missing))

    original_output_path = Registry.mapping["logger_mapping"]["path"].path
    Registry.mapping["logger_mapping"]["path"].path = str(source_output_dir)
    try:
        for ag in trainer_obj.agents:
            ag.load_model(checkpoint)
    finally:
        Registry.mapping["logger_mapping"]["path"].path = original_output_path


def save_eval_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    source_output_dir: Path,
    checkpoint: str,
    seed: int,
) -> None:
    metadata = {
        "seed": seed,
        "checkpoint": checkpoint,
        "source_output_dir": str(source_output_dir),
        "config": config,
    }
    with (output_dir / "eval_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)


def collect_metric_result(
    metric: Any,
    seed: int,
    network: str,
    prefix: str,
    checkpoint: str,
    output_dir: Path,
    log_file: str,
) -> Dict[str, Any]:
    return {
        "seed": seed,
        "network": network,
        "prefix": prefix,
        "checkpoint": checkpoint,
        "travel_time": float(metric.real_average_travel_time()),
        "reward": float(metric.rewards()),
        "queue": float(metric.queue()),
        "delay": float(metric.delay()),
        "throughput": int(metric.throughput()),
        "output_dir": str(output_dir),
        "log_file": log_file,
    }


def default_summary_path(args: argparse.Namespace) -> Path:
    return (
        REPO_ROOT
        / "data"
        / "output_data"
        / args.task
        / f"{args.world}_{args.agent}"
        / args.network
        / f"{args.prefix}_summary.csv"
    )


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cleanup_after_eval(trainer_obj: Any) -> None:
    try:
        if getattr(trainer_obj, "dataset", None) is not None:
            trainer_obj.dataset.finalize()
    except Exception:
        pass

    del trainer_obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_one_eval(
    args: argparse.Namespace,
    seed: int,
    prefix: str,
    source_output_dir: Path,
    source_hyperparams: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    from common.registry import Registry
    from utils.logger import build_config, setup_logging

    run_args = make_run_args(args, seed, prefix)
    config, duplicate_config = build_config(run_args)
    config = apply_eval_overrides(config, args, seed, source_hyperparams)

    configure_registry(config)
    reset_logging_handlers()
    logger = setup_logging(logging.DEBUG if args.debug else logging.INFO)

    logger.info("Evaluation config loaded from configs/%s/%s.yml", args.task, args.agent)
    logger.info("Target simulator config: configs/sim/%s.cfg", args.network)
    logger.info("Dataset backend: %s", args.dataset)
    logger.info("Source checkpoint directory: %s", source_output_dir / "model")
    if duplicate_config:
        logger.debug("Duplicate config overrides: %s", duplicate_config)

    trainer_cls = Registry.mapping["trainer_mapping"][config["command"]["task"]]
    trainer_obj = trainer_cls(logger)
    output_dir = Path(Registry.mapping["logger_mapping"]["path"].path).resolve()
    save_eval_metadata(output_dir, config, source_output_dir, args.checkpoint, seed)

    logger.info("Evaluation output directory: %s", output_dir)
    logger.info("Runtime dataset path: %s", getattr(trainer_obj.dataset, "path", "unknown"))
    load_agents_from_source(trainer_obj, args.checkpoint, source_output_dir)
    logger.info("Loaded checkpoint '%s' from %s", args.checkpoint, source_output_dir / "model")

    metric = trainer_obj.test(drop_load=True)
    result = collect_metric_result(
        metric,
        seed=seed,
        network=args.network,
        prefix=prefix,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        log_file=trainer_obj.log_file,
    )
    cleanup_after_eval(trainer_obj)
    reset_logging_handlers()
    return result


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    if args.cpu:
        args.ngpu = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

    setup_project_modules()

    source_output_dir = resolve_source_output_dir(args)
    if not source_output_dir.exists():
        raise FileNotFoundError(f"Source output directory does not exist: {source_output_dir}")

    source_hyperparams = None
    if not args.no_source_hyperparams:
        source_hyperparams = load_source_hyperparameters(source_output_dir)
        if source_hyperparams is None:
            print(
                f"[warn] No hyperparameters.json found in {source_output_dir}; "
                "falling back to current YAML config.",
                file=sys.stderr,
            )

    rows: List[Dict[str, Any]] = []
    multiple_seeds = len(seeds) > 1
    for seed in seeds:
        prefix = args.prefix
        if multiple_seeds and not args.single_prefix:
            prefix = f"{args.prefix}_seed{seed}"
        print(f"[eval] seed={seed} network={args.network} prefix={prefix}")
        rows.append(run_one_eval(args, seed, prefix, source_output_dir, source_hyperparams))

    summary_path = Path(args.summary_file).expanduser() if args.summary_file else default_summary_path(args)
    summary_path = summary_path.resolve()
    write_summary_csv(summary_path, rows)

    print("\nEvaluation summary:")
    for row in rows:
        print(
            "seed={seed} travel_time={travel_time:.4f} reward={reward:.4f} "
            "queue={queue:.4f} delay={delay:.4f} throughput={throughput} "
            "log={log_file}".format(**row)
        )
    print(f"\nSaved CSV summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
