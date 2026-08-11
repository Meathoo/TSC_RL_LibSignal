#!/usr/bin/env python3
"""Run a command after a matching process exits."""

import argparse
import os
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--match", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def matching_process_exists(pid, expected):
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        command = cmdline.read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except FileNotFoundError:
        return False
    return expected in command


def main():
    args = parse_args()
    print(
        f"Waiting for PID {args.pid} containing {args.match!r} to exit...",
        flush=True,
    )
    while matching_process_exists(args.pid, args.match):
        time.sleep(max(1.0, args.poll_seconds))

    print("Predecessor finished; starting: " + subprocess.list2cmdline(args.command), flush=True)
    completed = subprocess.run(args.command, check=False)
    print(f"Chained command exited with status {completed.returncode}", flush=True)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
