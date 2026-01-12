#!/usr/bin/env python3
"""Utility for reading dataset_logger_node outputs."""

from __future__ import annotations

import argparse
import math
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import rerun as rr
except Exception:
    rr = None


@dataclass(frozen=True)
class DatasetSample:
    timestamp_ns: int
    pose: Dict[str, Any]
    cmd_twist: Dict[str, float]
    image: Optional[np.ndarray]
    image_path: str
    image_format: str
    stored_encoding: str
    source_encoding: str


def load_session_meta(dataset_path: str) -> Dict[str, Any]:
    manifest_path = _resolve_manifest_path(dataset_path)
    dataset_dir = os.path.dirname(manifest_path)
    session_path = os.path.join(dataset_dir, "session.json")
    if not os.path.exists(session_path):
        return {}
    with open(session_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_samples(dataset_path: str, *, load_images: bool = True) -> Iterator[DatasetSample]:
    """
    Iterate dataset samples from a dataset directory or samples.jsonl path.
    """

    manifest_path = _resolve_manifest_path(dataset_path)
    dataset_dir = os.path.dirname(manifest_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"samples.jsonl not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_info = record.get("image") or {}
            image_rel_path = str(image_info.get("path", ""))
            image_path = os.path.join(dataset_dir, image_rel_path)
            image_format = str(image_info.get("format", ""))
            stored_encoding = str(image_info.get("stored_encoding", ""))
            source_encoding = str(image_info.get("source_encoding", ""))

            image = None
            if load_images:
                image = _load_image(image_path, image_format, stored_encoding)

            yield DatasetSample(
                timestamp_ns=int(record.get("timestamp_ns", 0)),
                pose=record.get("pose") or {},
                cmd_twist=record.get("cmd_twist") or {},
                image=image,
                image_path=image_path,
                image_format=image_format,
                stored_encoding=stored_encoding,
                source_encoding=source_encoding,
            )


def _resolve_manifest_path(dataset_path: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(dataset_path))
    if os.path.isdir(expanded):
        return os.path.join(expanded, "samples.jsonl")
    return expanded


def _load_image(path: str, image_format: str, stored_encoding: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    fmt = image_format.lower()
    if fmt == "npy":
        return np.load(path)

    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image at {path}")
        if stored_encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    if Image is not None:
        with Image.open(path) as image_handle:
            img = np.array(image_handle.convert("RGB"))
        if stored_encoding == "bgr8":
            img = img[:, :, ::-1]
        return img

    raise RuntimeError(
        "Reading jpg/png requires opencv-python or pillow; "
        "install one of them or log in npy format."
    )


def _format_pose(pose: Dict[str, Any]) -> str:
    pos = pose.get("position") or {}
    return f"({pos.get('x', 0):.3f}, {pos.get('y', 0):.3f}, {pos.get('z', 0):.3f})"


def _pose_position(pose: Dict[str, Any]) -> tuple[float, float, float]:
    pos = pose.get("position") or {}
    return (
        float(pos.get("x", 0.0)),
        float(pos.get("y", 0.0)),
        float(pos.get("z", 0.0)),
    )


def _xy_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _to_rgb(image: np.ndarray, stored_encoding: str) -> np.ndarray:
    if stored_encoding == "bgr8":
        return image[:, :, ::-1]
    return image


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Read dataset_logger_node outputs")
    parser.add_argument("dataset", help="Dataset directory or samples.jsonl path")
    parser.add_argument("--limit", type=int, default=5, help="Print first N samples")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable rerun visualization",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip loading images (faster for metadata inspection)",
    )
    args = parser.parse_args()

    show_enabled = not args.no_show
    if show_enabled:
        if rr is None:
            raise RuntimeError(
                "rerun-sdk is required for visualization; install with `uv pip install rerun-sdk`."
            )
        rr.init("tide_dataset", spawn=True)
        path_positions: list[tuple[float, float, float]] = []
        max_jump_m = 20.0
        last_good_position: Optional[tuple[float, float, float]] = None
        pending_jump: Optional[tuple[tuple[float, float, float], int, int]] = None
    else:
        path_positions = []

    for index, sample in enumerate(iter_samples(args.dataset, load_images=not args.no_images)):
        twist = sample.cmd_twist
        image_shape = tuple(sample.image.shape) if sample.image is not None else None
        print(
            f"[{index}] t={sample.timestamp_ns} "
            f"pos={_format_pose(sample.pose)} "
            f"cmd=({twist.get('vx', 0):.2f}, {twist.get('vy', 0):.2f}, {twist.get('wz', 0):.2f}) "
            f"image_shape={image_shape}"
        )
        if show_enabled:
            position = _pose_position(sample.pose)
            to_log: list[tuple[tuple[float, float, float], int, int]] = []
            if last_good_position is None and pending_jump is None:
                to_log.append((position, sample.timestamp_ns, index))
                last_good_position = position
            elif pending_jump is not None and last_good_position is not None:
                if _xy_distance(last_good_position, position) <= max_jump_m:
                    pending_jump = None
                    to_log.append((position, sample.timestamp_ns, index))
                    last_good_position = position
                else:
                    to_log.append(pending_jump)
                    last_good_position = pending_jump[0]
                    pending_jump = None
                    if _xy_distance(last_good_position, position) > max_jump_m:
                        pending_jump = (position, sample.timestamp_ns, index)
                    else:
                        to_log.append((position, sample.timestamp_ns, index))
                        last_good_position = position
            elif last_good_position is not None:
                if _xy_distance(last_good_position, position) > max_jump_m:
                    pending_jump = (position, sample.timestamp_ns, index)
                else:
                    to_log.append((position, sample.timestamp_ns, index))
                    last_good_position = position

            for pos, ts_ns, frame_idx in to_log:
                rr.set_time_sequence("frame", frame_idx)
                rr.set_time_nanos("timestamp", ts_ns)
                rr.log("pose/position", rr.Points3D([pos]))
                path_positions.append(pos)
                if len(path_positions) > 1:
                    rr.log(
                        "pose/path",
                        rr.LineStrips3D(
                            [np.array(path_positions, dtype=np.float32)],
                            colors=np.array([[0, 128, 255]], dtype=np.uint8),
                        ),
                    )

            rr.set_time_sequence("frame", index)
            rr.set_time_nanos("timestamp", sample.timestamp_ns)
            rr.log("cmd/vx", rr.Scalars(float(twist.get("vx", 0.0))))
            rr.log("cmd/vy", rr.Scalars(float(twist.get("vy", 0.0))))
            rr.log("cmd/wz", rr.Scalars(float(twist.get("wz", 0.0))))
            if sample.image is not None:
                rr.log(
                    "camera/image",
                    rr.Image(_to_rgb(sample.image, sample.stored_encoding)),
                )
        if args.limit and (index + 1) >= args.limit:
            break

    if show_enabled and pending_jump is not None:
        pos, ts_ns, frame_idx = pending_jump
        rr.set_time_sequence("frame", frame_idx)
        rr.set_time_nanos("timestamp", ts_ns)
        rr.log("pose/position", rr.Points3D([pos]))
        path_positions.append(pos)
        if len(path_positions) > 1:
            rr.log(
                "pose/path",
                rr.LineStrips3D(
                    [np.array(path_positions, dtype=np.float32)],
                    colors=np.array([[0, 128, 255]], dtype=np.uint8),
                ),
            )


if __name__ == "__main__":
    _cli()
