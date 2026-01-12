#!/usr/bin/env python3
"""Trim dataset_logger_node outputs to active motion windows."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from typing import Any, Dict, Iterator, Optional, Tuple


def _resolve_manifest_path(dataset_path: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(dataset_path))
    if os.path.isdir(expanded):
        return os.path.join(expanded, "samples.jsonl")
    return expanded


def _iter_records(manifest_path: str) -> Iterator[Dict[str, Any]]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _pose_xy(record: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pose = record.get("pose") or {}
    pos = pose.get("position") or {}
    try:
        return float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
    except (TypeError, ValueError):
        return None


def _linear_speed(record: Dict[str, Any]) -> float:
    twist = record.get("cmd_twist") or {}
    try:
        vx = float(twist.get("vx", 0.0))
        vy = float(twist.get("vy", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return math.hypot(vx, vy)


def _angular_speed(record: Dict[str, Any]) -> float:
    twist = record.get("cmd_twist") or {}
    try:
        return abs(float(twist.get("wz", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _scan_activity(
    manifest_path: str,
    *,
    min_linear: float,
    min_angular: float,
    min_translation: float,
) -> Tuple[list[bool], list[int]]:
    active_flags: list[bool] = []
    timestamps: list[int] = []
    last_pose: Optional[Tuple[float, float]] = None

    for record in _iter_records(manifest_path):
        active = False
        if _linear_speed(record) >= min_linear:
            active = True
        if _angular_speed(record) >= min_angular:
            active = True
        if min_translation > 0.0:
            pose = _pose_xy(record)
            if pose is not None and last_pose is not None:
                dx = pose[0] - last_pose[0]
                dy = pose[1] - last_pose[1]
                if math.hypot(dx, dy) >= min_translation:
                    active = True
            if pose is not None:
                last_pose = pose

        active_flags.append(active)
        try:
            timestamps.append(int(record.get("timestamp_ns", 0)))
        except (TypeError, ValueError):
            timestamps.append(0)

    return active_flags, timestamps


def _find_active_window(flags: list[bool], min_active_samples: int) -> Tuple[Optional[int], Optional[int]]:
    if not flags:
        return None, None

    run = 0
    start: Optional[int] = None
    for idx, active in enumerate(flags):
        run = run + 1 if active else 0
        if run >= min_active_samples:
            start = idx - min_active_samples + 1
            break

    run = 0
    end: Optional[int] = None
    for idx in range(len(flags) - 1, -1, -1):
        active = flags[idx]
        run = run + 1 if active else 0
        if run >= min_active_samples:
            end = idx
            break

    return start, end


def _prepare_output_dir(path: str, *, overwrite: bool) -> None:
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Output directory exists: {path}")
    os.makedirs(path, exist_ok=True)


def _copy_image(dataset_dir: str, output_dir: str, record: Dict[str, Any]) -> None:
    image = record.get("image") or {}
    rel_path = image.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        return

    if os.path.isabs(rel_path):
        source_path = rel_path
        rel_path = os.path.relpath(rel_path, dataset_dir)
        image["path"] = rel_path
    else:
        source_path = os.path.join(dataset_dir, rel_path)

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Image not found: {source_path}")

    dest_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(source_path, dest_path)


def _write_session(
    dataset_dir: str,
    output_dir: str,
    *,
    start_index: int,
    end_index: int,
    timestamps: list[int],
    args: argparse.Namespace,
) -> None:
    session_path = os.path.join(dataset_dir, "session.json")
    session: Dict[str, Any] = {}
    if os.path.exists(session_path):
        with open(session_path, "r", encoding="utf-8") as handle:
            session = json.load(handle)

    trim_meta = {
        "source": os.path.abspath(dataset_dir),
        "start_index": start_index,
        "end_index": end_index,
        "start_timestamp_ns": timestamps[start_index] if timestamps else 0,
        "end_timestamp_ns": timestamps[end_index] if timestamps else 0,
        "min_linear": args.min_linear,
        "min_angular": args.min_angular,
        "min_translation": args.min_translation,
        "min_active_samples": args.min_active_samples,
        "pad_samples": args.pad_samples,
    }
    session["trim"] = trim_meta

    out_session_path = os.path.join(output_dir, "session.json")
    with open(out_session_path, "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=2)
        handle.write("\n")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Trim dataset logs to active motion segments")
    parser.add_argument("dataset", help="Dataset directory or samples.jsonl path")
    parser.add_argument("--output", help="Output directory for trimmed dataset")
    parser.add_argument("--min-linear", type=float, default=0.05, help="Linear speed threshold (m/s)")
    parser.add_argument("--min-angular", type=float, default=0.1, help="Angular speed threshold (rad/s)")
    parser.add_argument(
        "--min-translation",
        type=float,
        default=0.0,
        help="Pose delta threshold (m) to mark motion active",
    )
    parser.add_argument(
        "--min-active-samples",
        type=int,
        default=3,
        help="Consecutive active samples required at start/end",
    )
    parser.add_argument(
        "--pad-samples",
        type=int,
        default=5,
        help="Extra samples before and after the active region",
    )
    parser.add_argument("--start-index", type=int, help="Manual start index override")
    parser.add_argument("--end-index", type=int, help="Manual end index override (inclusive)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print computed range and exit")
    args = parser.parse_args()

    manifest_path = _resolve_manifest_path(args.dataset)
    dataset_dir = os.path.dirname(manifest_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"samples.jsonl not found at {manifest_path}")

    flags, timestamps = _scan_activity(
        manifest_path,
        min_linear=args.min_linear,
        min_angular=args.min_angular,
        min_translation=args.min_translation,
    )
    auto_start, auto_end = _find_active_window(flags, max(1, args.min_active_samples))
    if auto_start is None or auto_end is None:
        raise RuntimeError("No active segment found; try lowering thresholds or use --start-index/--end-index.")

    auto_start = max(0, auto_start - max(0, args.pad_samples))
    auto_end = min(len(flags) - 1, auto_end + max(0, args.pad_samples))

    start_index = args.start_index if args.start_index is not None else auto_start
    end_index = args.end_index if args.end_index is not None else auto_end

    if start_index < 0 or end_index >= len(flags):
        raise ValueError(f"Trim range must be within [0, {len(flags) - 1}]")
    if start_index > end_index:
        raise ValueError("Start index must be <= end index")

    if args.dry_run:
        total = end_index - start_index + 1
        print(f"Trim range: {start_index}..{end_index} ({total} samples)")
        return

    output_dir = args.output or f"{dataset_dir}_trimmed"
    _prepare_output_dir(output_dir, overwrite=args.overwrite)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    output_manifest = os.path.join(output_dir, "samples.jsonl")
    kept = 0
    with open(output_manifest, "w", encoding="utf-8") as handle:
        for idx, record in enumerate(_iter_records(manifest_path)):
            if idx < start_index:
                continue
            if idx > end_index:
                break
            _copy_image(dataset_dir, output_dir, record)
            handle.write(json.dumps(record) + "\n")
            kept += 1

    _write_session(
        dataset_dir,
        output_dir,
        start_index=start_index,
        end_index=end_index,
        timestamps=timestamps,
        args=args,
    )

    print(f"Wrote {kept} samples to {output_dir}")


if __name__ == "__main__":
    _cli()
