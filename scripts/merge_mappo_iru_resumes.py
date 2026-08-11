#!/usr/bin/env python3
"""Merge MAPPO-IRU warm-resume artifacts back into their source run folders.

The script preserves the pre-merge log files as ``*.pre_resume_merge.bak``,
rebuilds the canonical DTL/PERF/BRF logs, copies continuation checkpoints, and
writes a manifest describing every source and decision.  Dataset files are not
merged because the on-the-fly LMDB stores are not needed for PPO continuation
or evaluation.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "data/output_data/tsc/cityflow_mappo_iru/cityflow16x3",
    )
    parser.add_argument("--resume-glob", default="*_resume250")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parsed_dtl_lines(path):
    parsed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            step = int(fields[2])
        except ValueError:
            continue
        parsed.append((line, fields[1], step, fields))
    return parsed


def select_source_dtl(source):
    candidates = []
    for path in sorted((source / "logger").glob("*_DTL.log")):
        parsed = parsed_dtl_lines(path)
        train_steps = [step for _, mode, step, _ in parsed if mode == "TRAIN"]
        if train_steps and min(train_steps) == 0:
            candidates.append((max(train_steps), path))
    if not candidates:
        raise FileNotFoundError(f"No episode-zero DTL log found in {source / 'logger'}")
    return max(candidates, key=lambda item: (item[0], item[1].name))[1]


def paired_log(dtl_path, kind):
    candidate = Path(str(dtl_path).replace("_DTL.log", f"_{kind}.log"))
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing matching {kind} log for {dtl_path}")
    return candidate


def atomic_write(path, text):
    temporary = path.with_name(path.name + ".merge_tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def backup_once(path):
    backup = path.with_name(path.name + ".pre_resume_merge.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def filter_tsv(path, before_episode):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty TSV log: {path}")
    header = lines[0]
    kept = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        try:
            step = int(fields[2])
        except ValueError:
            continue
        if step < before_episode:
            kept.append(line)
    return header, kept


def copy_or_verify(source, destination):
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise FileExistsError(
                f"Refusing to overwrite different file: {destination}"
            )
        return "verified"
    shutil.copy2(source, destination)
    return "copied"


def best_test(parsed):
    rows = []
    for _, mode, step, fields in parsed:
        if mode == "TEST" and len(fields) >= 4:
            rows.append((float(fields[3]), step))
    if not rows:
        raise ValueError("DTL log contains no TEST rows")
    return min(rows)


def merge_one(resume, dry_run=False):
    metadata_path = resume / "resume_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing resume metadata: {metadata_path}")
    resume_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = Path(resume_metadata["source_run"]).resolve()
    from_episode = int(resume_metadata["from_episode"])
    to_episode = int(resume_metadata["to_episode"])
    if not source.is_dir():
        raise FileNotFoundError(f"Missing source run: {source}")

    manifest_path = source / f"resume_merge_{from_episode}_{to_episode}.json"
    if manifest_path.exists():
        print(f"[skip] already merged: {source.name}")
        return

    source_dtl = select_source_dtl(source)
    resume_dtl_files = sorted((resume / "logger").glob("*_DTL.log"))
    if len(resume_dtl_files) != 1:
        raise ValueError(f"Expected one resume DTL in {resume / 'logger'}")
    resume_dtl = resume_dtl_files[0]
    source_perf = paired_log(source_dtl, "PERF")
    source_brf = paired_log(source_dtl, "BRF")
    resume_perf = paired_log(resume_dtl, "PERF")
    resume_brf = paired_log(resume_dtl, "BRF")

    source_parsed = parsed_dtl_lines(source_dtl)
    resume_parsed = parsed_dtl_lines(resume_dtl)
    source_prefix = [
        line for line, _, step, _ in source_parsed if step < from_episode
    ]
    resume_lines = [line for line, _, _, _ in resume_parsed]
    train_source = [
        step for _, mode, step, _ in source_parsed
        if mode == "TRAIN" and step < from_episode
    ]
    train_resume = [
        step for _, mode, step, _ in resume_parsed if mode == "TRAIN"
    ]
    if train_source != list(range(from_episode)):
        raise ValueError(
            f"Source TRAIN range is incomplete for {source.name}: "
            f"{train_source[:3]}...{train_source[-3:]}"
        )
    if train_resume != list(range(from_episode, to_episode)):
        raise ValueError(
            f"Resume TRAIN range is incomplete for {resume.name}: "
            f"{train_resume[:3]}...{train_resume[-3:]}"
        )

    combined_dtl = "\n".join(source_prefix + resume_lines) + "\n"
    source_perf_header, source_perf_rows = filter_tsv(source_perf, from_episode)
    resume_perf_lines = resume_perf.read_text(encoding="utf-8").splitlines()
    if not resume_perf_lines or resume_perf_lines[0] != source_perf_header:
        raise ValueError(f"PERF headers differ for {source.name}")
    combined_perf = (
        "\n".join([source_perf_header] + source_perf_rows + resume_perf_lines[1:])
        + "\n"
    )
    source_brf_backup = source_brf.with_name(
        source_brf.name + ".pre_resume_merge.bak"
    )
    source_brf_input = source_brf_backup if source_brf_backup.exists() else source_brf
    combined_brf = (
        source_brf_input.read_text(encoding="utf-8").rstrip()
        + "\n"
        + resume_brf.read_text(encoding="utf-8").lstrip()
    )
    if not combined_brf.endswith("\n"):
        combined_brf += "\n"

    original_best = best_test(
        [item for item in source_parsed if item[2] < from_episode]
    )
    resumed_best = best_test(resume_parsed)
    global_best_segment = (
        "resume" if resumed_best[0] <= original_best[0] else "original"
    )
    seed0_discontinuity = source.name.startswith("seed0_")

    print(
        f"[merge] {source.name}: TEST best "
        f"pre={original_best[0]:.4f}@{original_best[1]}, "
        f"resume={resumed_best[0]:.4f}@{resumed_best[1]}"
    )
    if dry_run:
        return

    backups = {
        "DTL": str(backup_once(source_dtl)),
        "PERF": str(backup_once(source_perf)),
        "BRF": str(backup_once(source_brf)),
    }
    atomic_write(source_dtl, combined_dtl)
    atomic_write(source_perf, combined_perf)
    atomic_write(source_brf, combined_brf)

    model_dir = source / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_actions = {}
    for checkpoint in sorted((resume / "model").glob("*.pt")):
        if checkpoint.name == "best_0.pt":
            continue
        checkpoint_episode = checkpoint.name.split("_", 1)[0]
        if checkpoint_episode == str(from_episode):
            # The copied boundary checkpoint is loaded first, then save_rate
            # writes episode `from_episode` again after one resumed update.
            # Keep the source run's boundary checkpoint as the true starting
            # point and skip this post-update name collision.
            model_actions[checkpoint.name] = "skipped_post_resume_update_collision"
            continue
        model_actions[checkpoint.name] = copy_or_verify(
            checkpoint, model_dir / checkpoint.name
        )

    source_best = model_dir / "best_0.pt"
    original_best_copy = model_dir / "best_ep000_099_0.pt"
    resume_best = resume / "model/best_0.pt"
    resumed_best_copy = model_dir / "best_ep100_249_0.pt"
    if source_best.is_file() and not original_best_copy.exists():
        shutil.copy2(source_best, original_best_copy)
    copy_or_verify(resume_best, resumed_best_copy)
    if global_best_segment == "resume":
        shutil.copy2(resume_best, source_best)

    copy_or_verify(
        resume / "hyperparameters.json",
        source / "hyperparameters_resume250.json",
    )
    copy_or_verify(
        metadata_path,
        source / "resume_metadata_250.json",
    )

    manifest = {
        "source_run": str(source),
        "resume_run": str(resume.resolve()),
        "from_episode": from_episode,
        "to_episode": to_episode,
        "canonical_logs": {
            "DTL": str(source_dtl),
            "PERF": str(source_perf),
            "BRF": str(source_brf),
        },
        "backups": backups,
        "source_dtl_sha256_after": sha256(source_dtl),
        "resume_dtl_sha256": sha256(resume_dtl),
        "model_actions": model_actions,
        "best": {
            "pre_resume": {
                "travel_time": original_best[0],
                "episode": original_best[1],
                "checkpoint": str(original_best_copy),
            },
            "resume": {
                "travel_time": resumed_best[0],
                "episode": resumed_best[1],
                "checkpoint": str(resumed_best_copy),
            },
            "global_segment": global_best_segment,
            "canonical_checkpoint": str(source_best),
        },
        "seed0_curve_warning": (
            "The available seed0 episode 0-99 log is from the first duplicated "
            "run, while the resumed checkpoint was overwritten by the second "
            "run. Treat the 0-99/100 boundary as discontinuous."
            if seed0_discontinuity
            else None
        ),
        "dataset_merge": "skipped (on-the-fly PPO dataset is not needed)",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    resumes = sorted(args.root.resolve().glob(args.resume_glob))
    if not resumes:
        raise FileNotFoundError(
            f"No resume directories matched {args.resume_glob!r} under {args.root}"
        )
    for resume in resumes:
        merge_one(resume, dry_run=args.dry_run)
    print(f"{'Validated' if args.dry_run else 'Merged'} {len(resumes)} run(s).")


if __name__ == "__main__":
    main()
