#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


METRICS = ["travel_time", "loss", "reward", "queue", "delay", "throughput"]
MODES = ["TRAIN", "TEST"]


@dataclass(frozen=True)
class SeedGroup:
    key: str
    label: str
    paths: List[str]


@dataclass
class Record:
    algo: str
    mode: str
    episode: int
    travel_time: float
    loss: float
    reward: float
    queue: float
    delay: float
    throughput: float


@dataclass
class LoadedSeed:
    group_key: str
    group_label: str
    seed_label: str
    log_path: Path
    records: List[Record]


@dataclass
class AvgSeries:
    group_key: str
    group_label: str
    mode: str
    metric: str
    episodes: List[int]
    mean: List[float]
    std: List[float]
    min_value: List[float]
    max_value: List[float]
    seed_count: List[int]


# Default groups mirror the two methods currently enabled in compare.py.
SEED_GROUPS: List[SeedGroup] = [
    SeedGroup(
        key="learned_queue",
        label="learned+queue",
        paths=[
            "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_learned64_mlp_queue_ep250/logger/2026_05_31-19_10_31_DTL.log",
            "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_learned64_mlp_queue_ep250/logger/2026_06_05-09_31_00_DTL.log",
            "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_learned64_mlp_queue_ep250/logger/2026_06_05-20_10_30_DTL.log",
        ],
    ),
    SeedGroup(
        key="learned_queuepress02",
        label="learned+queuePress0.2/0",
        paths=[
            "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_learned64_mlp_queuePress02_ep250/logger/2026_06_02-18_53_47_DTL.log",
            "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_learned64_mlp_queuePress02_ep250/logger/2026_06_04-19_14_08_DTL.log",
            "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_learned64_mlp_queuePress02_ep250/logger/2026_06_08-20_58_19_DTL.log",
        ],
    ),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def parse_line(line: str) -> Optional[Record]:
    parts = line.strip().split("\t")
    if len(parts) != 9:
        return None
    try:
        return Record(
            algo=parts[0],
            mode=parts[1],
            episode=int(parts[2]),
            travel_time=float(parts[3]),
            loss=float(parts[4]),
            reward=float(parts[5]),
            queue=float(parts[6]),
            delay=float(parts[7]),
            throughput=float(parts[8]),
        )
    except ValueError:
        return None


def load_log(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = parse_line(line)
            if record is not None:
                records.append(record)
    return records


def find_latest_dtl_log(folder: Path) -> Optional[Path]:
    pattern = str(folder / "**" / "*_DTL*.log")
    candidates = [Path(p) for p in glob.glob(pattern, recursive=True)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_path(raw_path: str) -> Optional[Path]:
    path = Path(raw_path).expanduser()
    root = repo_root()

    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
        as_posix = path.as_posix()
        for prefix in ("/DaRL/LibSignal/", "/home/meathoo/tsc_darl2/LibSignal/"):
            if as_posix.startswith(prefix):
                candidates.append(root / as_posix[len(prefix) :])
    else:
        candidates.append(root / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate if candidate.suffix == ".log" else None
        if candidate.is_dir():
            latest = find_latest_dtl_log(candidate)
            if latest is not None:
                return latest
    return None


def metric_value(record: Record, metric: str) -> float:
    return float(getattr(record, metric))


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= window:
        return values

    smoothed: List[float] = []
    half = window // 2
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        chunk = values[start:end]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def seed_series(
    records: List[Record],
    mode: str,
    metric: str,
    episode_start: Optional[int],
    episode_end: Optional[int],
    ma_window: int,
) -> Dict[int, float]:
    filtered = [r for r in records if r.mode == mode]
    filtered.sort(key=lambda r: r.episode)

    if episode_start is not None:
        filtered = [r for r in filtered if r.episode >= episode_start]
    if episode_end is not None:
        filtered = [r for r in filtered if r.episode <= episode_end]

    episodes = [r.episode for r in filtered]
    values = [metric_value(r, metric) for r in filtered]
    values = moving_average(values, ma_window)
    return dict(zip(episodes, values))


def aggregate_series(
    seeds: Sequence[LoadedSeed],
    mode: str,
    metric: str,
    episode_start: Optional[int],
    episode_end: Optional[int],
    ma_window: int,
    episode_policy: str,
) -> Optional[AvgSeries]:
    per_seed = [
        seed_series(seed.records, mode, metric, episode_start, episode_end, ma_window)
        for seed in seeds
    ]
    per_seed = [series for series in per_seed if series]
    if not per_seed:
        return None

    episode_sets = [set(series) for series in per_seed]
    if episode_policy == "union":
        episodes = sorted(set().union(*episode_sets))
    else:
        episodes = sorted(set.intersection(*episode_sets))
    if not episodes:
        return None

    means: List[float] = []
    stds: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []
    counts: List[int] = []

    for episode in episodes:
        values = [series[episode] for series in per_seed if episode in series]
        avg = sum(values) / len(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        means.append(avg)
        stds.append(math.sqrt(variance))
        mins.append(min(values))
        maxs.append(max(values))
        counts.append(len(values))

    first_seed = seeds[0]
    return AvgSeries(
        group_key=first_seed.group_key,
        group_label=first_seed.group_label,
        mode=mode,
        metric=metric,
        episodes=episodes,
        mean=means,
        std=stds,
        min_value=mins,
        max_value=maxs,
        seed_count=counts,
    )


def group_by_key(groups: Iterable[SeedGroup]) -> Dict[str, SeedGroup]:
    return {group.key: group for group in groups}


def select_groups(group_args: Optional[List[str]]) -> List[SeedGroup]:
    available = group_by_key(SEED_GROUPS)
    if not group_args or "all" in group_args:
        return list(SEED_GROUPS)

    selected: List[SeedGroup] = []
    unknown: List[str] = []
    for key in group_args:
        group = available.get(key)
        if group is None:
            unknown.append(key)
        else:
            selected.append(group)

    if unknown:
        valid = ", ".join(["all"] + sorted(available))
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}. Valid choices: {valid}")
    return selected


def select_values(raw_values: Optional[List[str]], valid_values: Sequence[str], name: str) -> List[str]:
    if not raw_values or "all" in raw_values:
        return list(valid_values)
    invalid = [value for value in raw_values if value not in valid_values]
    if invalid:
        valid = ", ".join(["all"] + list(valid_values))
        raise ValueError(f"Unknown {name}: {', '.join(invalid)}. Valid choices: {valid}")
    return raw_values


def load_groups(groups: Sequence[SeedGroup], strict: bool) -> Dict[str, List[LoadedSeed]]:
    loaded: Dict[str, List[LoadedSeed]] = {}
    for group in groups:
        group_seeds: List[LoadedSeed] = []
        for idx, raw_path in enumerate(group.paths):
            log_path = resolve_path(raw_path)
            seed_label = f"seed{idx}"
            if log_path is None:
                message = f"[WARN] Missing log for {group.key}/{seed_label}: {raw_path}"
                if strict:
                    raise FileNotFoundError(message)
                print(message)
                continue

            records = load_log(log_path)
            if not records:
                message = f"[WARN] Empty or unparsable log for {group.key}/{seed_label}: {log_path}"
                if strict:
                    raise RuntimeError(message)
                print(message)
                continue

            print(f"[OK] {group.label}/{seed_label} <- {log_path} (rows={len(records)})")
            group_seeds.append(
                LoadedSeed(
                    group_key=group.key,
                    group_label=group.label,
                    seed_label=seed_label,
                    log_path=log_path,
                    records=records,
                )
            )

        if group_seeds:
            loaded[group.key] = group_seeds
        elif strict:
            raise RuntimeError(f"No usable seeds loaded for group: {group.key}")
    return loaded


def build_all_series(
    loaded: Dict[str, List[LoadedSeed]],
    modes: Sequence[str],
    metrics: Sequence[str],
    episode_start: Optional[int],
    episode_end: Optional[int],
    ma_window: int,
    episode_policy: str,
) -> Dict[Tuple[str, str, str], AvgSeries]:
    result: Dict[Tuple[str, str, str], AvgSeries] = {}
    for group_key, seeds in loaded.items():
        for mode in modes:
            for metric in metrics:
                series = aggregate_series(
                    seeds=seeds,
                    mode=mode,
                    metric=metric,
                    episode_start=episode_start,
                    episode_end=episode_end,
                    ma_window=ma_window,
                    episode_policy=episode_policy,
                )
                if series is not None:
                    result[(group_key, mode, metric)] = series
    return result


def band_values(series: AvgSeries, band: str) -> Optional[Tuple[List[float], List[float]]]:
    if band == "none":
        return None
    if band == "minmax":
        return series.min_value, series.max_value
    lower = [avg - std for avg, std in zip(series.mean, series.std)]
    upper = [avg + std for avg, std in zip(series.mean, series.std)]
    return lower, upper


def apply_axis_style(ax, metric: str) -> None:
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25, linestyle="--")
    if metric == "loss":
        ax.set_yscale("symlog", linthresh=1.0)


def plot_metric(
    series_by_key: Dict[Tuple[str, str, str], AvgSeries],
    loaded: Dict[str, List[LoadedSeed]],
    mode: str,
    metric: str,
    output_dir: Path,
    dpi: int,
    band: str,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    used = 0

    for group_key in loaded:
        series = series_by_key.get((group_key, mode, metric))
        if series is None:
            continue
        line = ax.plot(series.episodes, series.mean, linewidth=1.8, label=series.group_label)[0]
        values = band_values(series, band)
        if values is not None:
            lower, upper = values
            ax.fill_between(series.episodes, lower, upper, color=line.get_color(), alpha=0.14)
        used += 1

    if used == 0:
        plt.close(fig)
        return

    ax.set_title(f"{metric.upper()} ({mode}) seed average")
    apply_axis_style(ax, metric)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"avg_metric_{metric}_{mode}.png"
    fig.savefig(out_file, dpi=dpi)
    print(f"[DONE] Saved: {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def plot_all_metrics(
    series_by_key: Dict[Tuple[str, str, str], AvgSeries],
    loaded: Dict[str, List[LoadedSeed]],
    mode: str,
    metrics: Sequence[str],
    output_dir: Path,
    dpi: int,
    band: str,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    any_data = False

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        if metric not in metrics:
            ax.set_title(f"{metric} (skipped)")
            ax.axis("off")
            continue

        used = 0
        for group_key in loaded:
            series = series_by_key.get((group_key, mode, metric))
            if series is None:
                continue
            line = ax.plot(series.episodes, series.mean, linewidth=1.3, label=series.group_label)[0]
            values = band_values(series, band)
            if values is not None:
                lower, upper = values
                ax.fill_between(series.episodes, lower, upper, color=line.get_color(), alpha=0.12)
            used += 1

        if used:
            any_data = True
            ax.set_title(metric)
            apply_axis_style(ax, metric)
        else:
            ax.set_title(f"{metric} (no data)")
            ax.axis("off")

    if not any_data:
        plt.close(fig)
        return

    handles, labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))

    fig.suptitle(f"{mode} seed-average comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"avg_all_metrics_{mode}.png"
    fig.savefig(out_file, dpi=dpi)
    print(f"[DONE] Saved: {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def write_summary_csv(
    series_by_key: Dict[Tuple[str, str, str], AvgSeries],
    loaded: Dict[str, List[LoadedSeed]],
    modes: Sequence[str],
    metrics: Sequence[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        summary_file = output_dir / f"avg_summary_{mode}.csv"
        series_file = output_dir / f"avg_series_{mode}.csv"

        with summary_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "group",
                    "metric",
                    "episodes",
                    "seed_count_min",
                    "seed_count_max",
                    "first_mean",
                    "last_mean",
                    "best_mean",
                    "best_episode",
                    "mean_of_mean",
                    "last_std",
                ]
            )

            for group_key in loaded:
                for metric in metrics:
                    series = series_by_key.get((group_key, mode, metric))
                    if series is None:
                        writer.writerow([group_key, metric, 0, "", "", "", "", "", "", "", ""])
                        continue

                    smaller_better = metric in {"travel_time", "loss", "queue", "delay"}
                    best_value = min(series.mean) if smaller_better else max(series.mean)
                    best_index = series.mean.index(best_value)
                    writer.writerow(
                        [
                            series.group_label,
                            metric,
                            len(series.episodes),
                            min(series.seed_count),
                            max(series.seed_count),
                            series.mean[0],
                            series.mean[-1],
                            best_value,
                            series.episodes[best_index],
                            sum(series.mean) / len(series.mean),
                            series.std[-1],
                        ]
                    )
        print(f"[DONE] Saved: {summary_file}")

        with series_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "metric", "episode", "mean", "std", "min", "max", "seed_count"])
            for group_key in loaded:
                for metric in metrics:
                    series = series_by_key.get((group_key, mode, metric))
                    if series is None:
                        continue
                    for idx, episode in enumerate(series.episodes):
                        writer.writerow(
                            [
                                series.group_label,
                                metric,
                                episode,
                                series.mean[idx],
                                series.std[idx],
                                series.min_value[idx],
                                series.max_value[idx],
                                series.seed_count[idx],
                            ]
                        )
        print(f"[DONE] Saved: {series_file}")


def print_groups() -> None:
    print("Available groups:")
    for group in SEED_GROUPS:
        print(f"  {group.key}: {group.label} ({len(group.paths)} seeds)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average multiple seed logs by selected groups and plot comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list-groups", action="store_true", help="Print available seed groups and exit.")
    parser.add_argument("--groups", nargs="+", default=["all"], help="Groups to plot. Use 'all' for every group.")
    parser.add_argument("--metrics", nargs="+", default=["all"], help="Metrics to plot. Use 'all' for every metric.")
    parser.add_argument("--modes", nargs="+", default=["TRAIN", "TEST"], help="Modes to plot.")
    parser.add_argument("--episode-start", type=int, default=0, help="First episode to include.")
    parser.add_argument("--episode-end", type=int, default=250, help="Last episode to include.")
    parser.add_argument("--ma-window", type=int, default=10, help="Moving-average window. Use 1 to disable.")
    parser.add_argument(
        "--episode-policy",
        choices=["intersection", "union"],
        default="intersection",
        help="How to align episodes across seeds.",
    )
    parser.add_argument(
        "--band",
        choices=["std", "minmax", "none"],
        default="std",
        help="Shaded band around each averaged line.",
    )
    parser.add_argument("--output-dir", default="avg_compare_outputs", help="Directory for plots and CSV files.")
    parser.add_argument("--dpi", type=int, default=170, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively after saving.")
    parser.add_argument("--no-all-plot", action="store_true", help="Skip 2x3 all-metrics figures.")
    parser.add_argument("--no-individual-plots", action="store_true", help="Skip one-figure-per-metric plots.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing or empty seed logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_groups:
        print_groups()
        return

    groups = select_groups(args.groups)
    metrics = select_values(args.metrics, METRICS, "metric")
    modes = select_values(args.modes, MODES, "mode")
    output_dir = Path(args.output_dir)

    loaded = load_groups(groups, strict=args.strict)
    if not loaded:
        raise RuntimeError("No usable seed logs were loaded.")

    series_by_key = build_all_series(
        loaded=loaded,
        modes=modes,
        metrics=metrics,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        ma_window=args.ma_window,
        episode_policy=args.episode_policy,
    )
    if not series_by_key:
        raise RuntimeError("No averaged series could be built for the selected inputs.")

    for mode in modes:
        if not args.no_all_plot:
            plot_all_metrics(
                series_by_key=series_by_key,
                loaded=loaded,
                mode=mode,
                metrics=metrics,
                output_dir=output_dir,
                dpi=args.dpi,
                band=args.band,
                show=args.show,
            )
        if not args.no_individual_plots:
            for metric in metrics:
                plot_metric(
                    series_by_key=series_by_key,
                    loaded=loaded,
                    mode=mode,
                    metric=metric,
                    output_dir=output_dir,
                    dpi=args.dpi,
                    band=args.band,
                    show=args.show,
                )

    write_summary_csv(
        series_by_key=series_by_key,
        loaded=loaded,
        modes=modes,
        metrics=metrics,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
