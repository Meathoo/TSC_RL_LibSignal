#!/usr/bin/env python3
"""Run the fixed-compute MAPPO-IRU experiment matrix sequentially."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run MAPPO-IRU steps/seeds and optional actor/critic ablations.',
    )
    parser.add_argument('--network', default='cityflow4x4')
    parser.add_argument('--world', default='cityflow', choices=['cityflow', 'sumo'])
    parser.add_argument('--ngpu', default='0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1,2])
    parser.add_argument('--iru-steps', nargs='+', type=int, default=[1, 2, 5])
    parser.add_argument(
        '--ablations',
        nargs='+',
        choices=['both', 'actor', 'critic'],
        default=['both'],
        help='both is the primary experiment; actor/critic are follow-up ablations.',
    )
    parser.add_argument('--prefix', default='mappo_iru')
    parser.add_argument(
        '--include-baseline',
        action='store_true',
        help='also run the unchanged MLP MAPPO baseline with performance profiling.',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    return parser.parse_args()


def architecture_pair(ablation):
    if ablation == 'actor':
        return 'iru', 'mlp'
    if ablation == 'critic':
        return 'mlp', 'iru'
    return 'iru', 'iru'


def run_command(command, repo_root, args):
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(
        command,
        cwd=repo_root,
        check=not args.continue_on_error,
    )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    if any(step <= 0 for step in args.iru_steps):
        raise ValueError('All --iru-steps values must be positive')

    if args.include_baseline:
        for seed in args.seeds:
            command = [
                python,
                '-u',
                'run.py',
                '--task',
                'tsc',
                '--agent',
                'mappo',
                '--world',
                args.world,
                '--network',
                args.network,
                '--prefix',
                f'{args.prefix}_mlp_seed{seed}',
                '--seed',
                str(seed),
                '--ngpu',
                args.ngpu,
                '--profile_performance',
                'true',
                '--native_use_agent_id',
                'False',
            ]
            run_command(command, repo_root, args)

    for ablation in args.ablations:
        actor_arch, value_arch = architecture_pair(ablation)
        for steps in args.iru_steps:
            for seed in args.seeds:
                command = [
                    python,
                    '-u',
                    'run.py',
                    '--task',
                    'tsc',
                    '--agent',
                    'mappo_iru',
                    '--world',
                    args.world,
                    '--network',
                    args.network,
                    '--prefix',
                    f'{args.prefix}_{ablation}_n{steps}_seed{seed}',
                    '--seed',
                    str(seed),
                    '--ngpu',
                    args.ngpu,
                    '--native_actor_arch',
                    actor_arch,
                    '--native_value_arch',
                    value_arch,
                    '--iru_steps',
                    str(steps),
                    '--profile_performance',
                    'true',
                    '--native_use_agent_id',
                    'False',
                ]
                run_command(command, repo_root, args)


if __name__ == '__main__':
    main()
