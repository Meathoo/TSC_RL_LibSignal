#!/usr/bin/env python3
"""Summarize LibSignal DTL logs in a Table-7-style best-test row.

The DTL log format is tab-separated:
agent, split, episode, travel_time, loss, reward, queue, delay, throughput
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Row:
    source: Path
    agent: str
    split: str
    episode: int
    travel_time: float
    loss: float
    reward: float
    queue: float
    delay: float
    throughput: int


def parse_log(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_no, fields in enumerate(reader, start=1):
            if not fields or all(not field.strip() for field in fields):
                continue
            if len(fields) != 9:
                raise ValueError(f"{path}:{line_no}: expected 9 tab-separated fields, got {len(fields)}")
            agent, split, episode, travel_time, loss, reward, queue, delay, throughput = fields
            rows.append(
                Row(
                    source=path,
                    agent=agent,
                    split=split.upper(),
                    episode=int(episode),
                    travel_time=float(travel_time),
                    loss=float(loss),
                    reward=float(reward),
                    queue=float(queue),
                    delay=float(delay),
                    throughput=int(float(throughput)),
                )
            )
    return rows


def best_row(rows: Iterable[Row], split: str, max_episode: int | None) -> Row:
    candidates = [
        row
        for row in rows
        if row.split == split and (max_episode is None or row.episode <= max_episode)
    ]
    if not candidates:
        cutoff = "" if max_episode is None else f" up to episode {max_episode}"
        raise ValueError(f"no {split} rows found{cutoff}")
    return min(candidates, key=lambda row: (row.travel_time, row.episode))


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values)


def fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def print_rows(rows: list[Row]) -> None:
    print("source\tagent\tsplit\tepisode\ttravel_time\tqueue\tdelay\tthroughput\treward")
    for row in rows:
        print(
            "\t".join(
                [
                    str(row.source),
                    row.agent,
                    row.split,
                    str(row.episode),
                    fmt(row.travel_time),
                    fmt(row.queue),
                    fmt(row.delay),
                    fmt(row.throughput),
                    fmt(row.reward),
                ]
            )
        )


def print_aggregate(rows: list[Row]) -> None:
    if len(rows) < 2:
        return
    metrics = {
        "travel_time": [row.travel_time for row in rows],
        "queue": [row.queue for row in rows],
        "delay": [row.delay for row in rows],
        "throughput": [float(row.throughput) for row in rows],
        "reward": [row.reward for row in rows],
    }
    print()
    print("metric\tmean\tstd")
    for name, values in metrics.items():
        print(f"{name}\t{statistics.mean(values):.4f}\t{sample_std(values):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Table-7-style best TEST metrics from LibSignal DTL logs."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Path(s) to *_DTL.log files.")
    parser.add_argument(
        "--split",
        default="TEST",
        choices=["TRAIN", "TEST"],
        help="Which split to summarize. Default: TEST.",
    )
    parser.add_argument(
        "--max-episode",
        type=int,
        default=None,
        help="Only consider rows with episode <= this value, e.g. 200 for paper-style cutoff.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not print mean/std when multiple logs are provided.",
    )
    args = parser.parse_args()

    best_rows: list[Row] = []
    for log_path in args.logs:
        rows = parse_log(log_path)
        best_rows.append(best_row(rows, args.split, args.max_episode))

    print_rows(best_rows)
    if not args.no_aggregate:
        print_aggregate(best_rows)


if __name__ == "__main__":
    main()
