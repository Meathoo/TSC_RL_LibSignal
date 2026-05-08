#!/usr/bin/env python3
import glob
import os
import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

METRICS = ["travel_time", "loss", "reward", "queue", "delay", "throughput"]
MODES = ["TRAIN", "TEST"]


# ========================
# User configuration
# ========================
# Support both folders and direct *_DTL.log files.
INPUT_PATHS = [
    # 4x4
    "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_ppo/cityflow4x4/hyperlight_ppo_seed0/logger/2026_05_02-10_53_38_DTL.log",
    # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/hyperlight_mappo_seed0/logger/2026_05_02-12_05_45_DTL.log",
    # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed1/logger/2026_05_03-16_56_25_DTL.log",
    # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed2/logger/2026_05_04-17_22_44_DTL.log",
    # "/DaRL/LibSignal/data/output_data/tsc/cityflow_colight/cityflow4x4/seed0/remote.log",
    "/DaRL/LibSignal/data/output_data/tsc/cityflow_colight/cityflow4x4/test/logger/2026_04_23-19_39_43_DTL.log",
    "/DaRL/LibSignal/data/output_data/tsc/cityflow_frap/cityflow4x4/seed0/logger/2026_05_05-20_56_37_DTL.log",
    "/DaRL/LibSignal/data/output_data/tsc/cityflow_mplight/cityflow4x4/seed0/logger/2026_05_06-02_38_21_DTL.log",

    # 7x28
    # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0/logger/2026_05_02-18_15_03_DTL.log",
    # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1/logger/2026_05_03-04_04_05_DTL.log",
    # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2/logger/2026_05_04-02_12_01_DTL.log",
    # "/DaRL/LibSignal/data/output_data/tsc/cityflow_colight/cityflow_7x28/test/logger/from_remote.log",
    # "/DaRL/LibSignal/data/output_data/tsc/cityflow_colight/cityflow7x28/seed2/logger/2026_05_04-18_17_36_DTL.log",
    # "/DaRL/LibSignal/data/output_data/tsc/cityflow_mplight/cityflow7x28/seed0/logger/2026_05_07-17_23_53_DTL.log",
]

# Optional: set custom labels for each input path.
# Keep empty to use folder/file names automatically.
DISPLAY_NAMES: List[str] = [
    # 4x4
    "hyperlight_ppo (seed0)",
    # "hyperlight_mappo (seed0)",
    # "hyperlight_mappo (seed1)",
    # "hyperlight_mappo (seed2)",
    # "colight",
    "colight (seed2)",
    "frap",
    "mplight"



    # 7x28
#     "hyperlight_mappo (seed0)",
#     "hyperlight_mappo (seed1)",
#     "hyperlight_mappo (seed2)",
#     "colight",
#     "colight (seed2)",
#     "mplight",
]

# Which modes to export in one run.
MODES_TO_PLOT = ["TRAIN", "TEST"]

# Moving Average Settings
USE_MOVING_AVERAGE = True  # Set to True to enable moving average smoothing
MOVING_AVERAGE_WINDOW = 5  # Window size for moving average (higher = smoother curve)

# Output settings
OUTPUT_DIR = "compare_outputs"
FIG_DPI = 170
SHOW_PLOT = False


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


def load_log(path: str) -> List[Record]:
    records: List[Record] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = parse_line(line)
            if r is not None:
                records.append(r)
    return records


def metric_value(record: Record, metric: str) -> float:
    return float(getattr(record, metric))


def apply_moving_average(values: List[float], window: int) -> List[float]:
    """
    Apply moving average smoothing to a list of values.
    
    Args:
        values: Input values to smooth
        window: Window size for moving average
    
    Returns:
        Smoothed values (same length as input)
    """
    if window <= 1 or len(values) <= window:
        return values
    
    smoothed = []
    for i in range(len(values)):
        # Center window around current index when possible
        start_idx = max(0, i - window // 2)
        end_idx = min(len(values), i + window // 2 + 1)
        window_values = values[start_idx:end_idx]
        smoothed.append(sum(window_values) / len(window_values))
    
    return smoothed


def find_latest_dtl_log(folder: str) -> Optional[str]:
    pattern = os.path.join(folder, "**", "*_DTL.log")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def resolve_log_path(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path if path.endswith(".log") else None
    if os.path.isdir(path):
        return find_latest_dtl_log(path)
    return None


def records_to_series(records: List[Record], mode: str, metric: str) -> Tuple[List[int], List[float]]:
    filtered = [r for r in records if r.mode == mode]
    filtered.sort(key=lambda r: r.episode)
    episodes = [r.episode for r in filtered]
    values = [metric_value(r, metric) for r in filtered]
    
    # Apply moving average if enabled
    if USE_MOVING_AVERAGE and len(values) > 1:
        values = apply_moving_average(values, MOVING_AVERAGE_WINDOW)
    
    return episodes, values


def resolve_labels(paths: List[str], names: List[str]) -> List[str]:
    if names:
        if len(names) != len(paths):
            raise ValueError(
                f"DISPLAY_NAMES 數量({len(names)})必須和 INPUT_PATHS 數量({len(paths)})一致。"
            )
        return names
    labels = []
    for path in paths:
        labels.append(os.path.basename(os.path.normpath(path)) or path)
    return labels


def load_sources(paths: List[str], labels: List[str]) -> List[Tuple[str, str, List[Record]]]:
    sources: List[Tuple[str, str, List[Record]]] = []
    for path, label in zip(paths, labels):
        log_path = resolve_log_path(path)
        if log_path is None:
            print(f"[WARN] 無法解析路徑(需為資料夾或 .log): {path}")
            continue
        records = load_log(log_path)
        if not records:
            print(f"[WARN] 無可解析資料: {log_path}")
            continue
        print(f"[OK] {label} <- {log_path} (rows={len(records)})")
        sources.append((label, log_path, records))
    return sources


def plot_all_metrics(
    sources: List[Tuple[str, str, List[Record]]],
    mode: str,
    output_dir: str,
    show: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    any_data = False

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        metric_used = 0
        for label, _log_path, records in sources:
            x, y = records_to_series(records, mode=mode, metric=metric)
            if not x:
                continue
            ax.plot(x, y, linewidth=1.7, label=label)
            metric_used += 1

        if metric_used > 0:
            any_data = True
            ax.set_title(metric)
            ax.set_xlabel("Episode")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
            if metric == "loss":
                # Prevent exploding-loss curves from flattening other lines.
                ax.set_yscale("symlog", linthresh=1.0)
        else:
            ax.set_title(f"{metric} (no data)")
            ax.axis("off")

    if not any_data:
        plt.close(fig)
        raise RuntimeError(f"mode={mode} 沒有可繪製資料。")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    
    title = f"{mode} - all metrics comparison"
    if USE_MOVING_AVERAGE:
        title += f" (MA window={MOVING_AVERAGE_WINDOW})"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"all_metrics_{mode}.png")
    fig.savefig(out_file, dpi=FIG_DPI)
    print(f"[DONE] 圖已輸出: {out_file}")

    if show:
        plt.show()
    plt.close(fig)


def plot_individual_metrics(
    sources: List[Tuple[str, str, List[Record]]],
    mode: str,
    output_dir: str,
    show: bool,
) -> None:
    """Plot each metric in a separate figure."""
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_used = 0
        
        for label, _log_path, records in sources:
            x, y = records_to_series(records, mode=mode, metric=metric)
            if not x:
                continue
            ax.plot(x, y, linewidth=2.0, label=label, marker='o', markersize=3, alpha=0.8)
            metric_used += 1
        
        if metric_used > 0:
            ax.set_title(f"{metric.upper()} ({mode})", fontsize=14, pad=15)
            ax.set_xlabel("Episode", fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(alpha=0.3, linestyle='--')
            
            if metric == "loss":
                ax.set_yscale("symlog", linthresh=1.0)
            
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="best", fontsize=11)
            
            fig.tight_layout()
            
            os.makedirs(output_dir, exist_ok=True)
            out_file = os.path.join(output_dir, f"metric_{metric}_{mode}.png")
            fig.savefig(out_file, dpi=FIG_DPI)
            print(f"[DONE] 圖已輸出: {out_file}")
        
        if show:
            plt.show()
        plt.close(fig)


def build_summary_csv(sources: List[Tuple[str, str, List[Record]]], mode: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"summary_{mode}.csv")

    header = ["name", "metric", "count", "first", "last", "best", "mean"]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for label, _log_path, records in sources:
            mode_records = [r for r in records if r.mode == mode]
            mode_records.sort(key=lambda r: r.episode)
            for metric in METRICS:
                vals = [metric_value(r, metric) for r in mode_records]
                if not vals:
                    writer.writerow([label, metric, 0, "", "", "", ""])
                    continue

                smaller_better = metric in {"travel_time", "loss", "queue", "delay"}
                best = min(vals) if smaller_better else max(vals)
                avg = sum(vals) / len(vals)
                writer.writerow([
                    label,
                    metric,
                    len(vals),
                    vals[0],
                    vals[-1],
                    best,
                    avg,
                ])

    print(f"[DONE] 表格已輸出: {out_file}")


def main():
    labels = resolve_labels(INPUT_PATHS, DISPLAY_NAMES)
    sources = load_sources(INPUT_PATHS, labels)
    if not sources:
        raise RuntimeError("沒有可用資料來源，請檢查 INPUT_PATHS。")

    for mode in MODES_TO_PLOT:
        if mode not in MODES:
            print(f"[WARN] 跳過未知 mode: {mode}")
            continue
        plot_all_metrics(
            sources=sources,
            mode=mode,
            output_dir=OUTPUT_DIR,
            show=SHOW_PLOT,
        )
        plot_individual_metrics(
            sources=sources,
            mode=mode,
            output_dir=OUTPUT_DIR,
            show=SHOW_PLOT,
        )
        build_summary_csv(
            sources=sources,
            mode=mode,
            output_dir=OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()
