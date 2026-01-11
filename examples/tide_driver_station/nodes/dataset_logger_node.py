#!/usr/bin/env python3
"""
DatasetLoggerNode: logs robot pose, camera images, and commanded twist for
end-to-end navigation training.

Output structure (default):
  logs/datasets/<robot_id>_<timestamp>/
    samples.jsonl
    images/
      <timestamp_ns>.<ext>

Each JSON line includes:
- timestamp_ns
- pose (position + orientation)
- cmd_twist (vx, vy, wz)
- image metadata (path, height, width, encoding)
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from tide.core.node import BaseNode


def _parse_twist(msg: Any) -> Optional[Tuple[float, float, float]]:
    try:
        if not isinstance(msg, dict):
            return None
        lin = msg.get("linear") or {}
        vx = float(lin.get("x", 0.0))
        vy = float(lin.get("y", 0.0))
        wz = float(msg.get("angular", msg.get("w", 0.0)))
        return vx, vy, float(wz)
    except Exception:
        return None


def _parse_pose(msg: Any) -> Optional[Dict[str, Any]]:
    try:
        if not isinstance(msg, dict):
            return None
        pos = msg.get("position") or {}
        ori = msg.get("orientation") or {}
        pose = {
            "position": {
                "x": float(pos.get("x", 0.0)),
                "y": float(pos.get("y", 0.0)),
                "z": float(pos.get("z", 0.0)),
            },
            "orientation": {
                "w": float(ori.get("w", 1.0)),
                "x": float(ori.get("x", 0.0)),
                "y": float(ori.get("y", 0.0)),
                "z": float(ori.get("z", 0.0)),
            },
        }
        header = msg.get("header")
        if isinstance(header, dict):
            frame_id = header.get("frame_id")
            if isinstance(frame_id, str) and frame_id:
                pose["frame_id"] = frame_id
        return pose
    except Exception:
        return None


def _normalize_output_path(path: str) -> str:
    normalized = os.path.expanduser(os.path.expandvars(path))
    if "\\ " in normalized:
        candidate = normalized.replace("\\ ", " ")
        if os.path.exists(candidate) or os.path.exists(os.path.dirname(candidate)):
            print(
                "DatasetLogger: output path looks shell-escaped; "
                f"using '{candidate}'"
            )
            return candidate
    return normalized


class DatasetLoggerNode(BaseNode):
    GROUP = ""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        params = config or {}

        self.hz = float(params.get("update_rate", params.get("hz", 20.0)))
        self.image_topic = str(params.get("image_topic", "sensor/camera/front/image"))
        self.pose_topic = str(params.get("pose_topic", "state/pose3d"))
        self.twist_topic = str(params.get("twist_topic", "cmd/twist"))
        self.max_age_s = float(params.get("max_age_s", 0.5))
        self.flush_every = int(params.get("flush_every", 1))
        self.convert_rgb = bool(params.get("convert_rgb", True))
        self.image_format = str(params.get("image_format", "npy")).lower()
        self.jpeg_quality = int(params.get("jpeg_quality", 90))

        output_dir = params.get("output_dir")
        output_root = str(params.get("output_root", "logs/datasets"))
        if output_dir:
            output_dir = _normalize_output_path(str(output_dir))
        else:
            output_root = _normalize_output_path(output_root)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_root, f"{self.ROBOT_ID}_{stamp}")
        self.output_dir = str(output_dir)
        self.images_dir = os.path.join(self.output_dir, "images")
        try:
            os.makedirs(self.images_dir, exist_ok=True)
        except PermissionError as exc:
            raise PermissionError(
                "DatasetLogger: cannot create output directory "
                f"'{self.images_dir}'. Check output_root/output_dir permissions."
            ) from exc

        if self.image_format not in ("npy", "jpg", "jpeg", "png"):
            print(f"DatasetLogger: unknown image_format '{self.image_format}', using npy")
            self.image_format = "npy"
        if self.image_format in ("jpg", "jpeg", "png") and cv2 is None:
            print("DatasetLogger: cv2 not available; falling back to .npy images")
            self.image_format = "npy"

        self._manifest_path = os.path.join(self.output_dir, "samples.jsonl")
        self._manifest = open(self._manifest_path, "a", encoding="utf-8")
        self._record_count = 0

        self._last_img: Optional[Dict[str, Any]] = None
        self._last_img_ts_ns: int = 0
        self._img_seq: int = 0
        self._last_logged_seq: int = 0

        self._last_pose: Optional[Dict[str, Any]] = None
        self._last_twist: Optional[Tuple[float, float, float]] = None
        self._t_pose: float = 0.0
        self._t_twist: float = 0.0

        self.subscribe(self.image_topic, self._on_image)
        self.subscribe(self.pose_topic, self._on_pose)
        self.subscribe(self.twist_topic, self._on_twist)

        session_meta = {
            "robot_id": self.ROBOT_ID,
            "start_time_ns": time.time_ns(),
            "image_topic": self.image_topic,
            "pose_topic": self.pose_topic,
            "twist_topic": self.twist_topic,
            "image_format": self.image_format,
        }
        session_path = os.path.join(self.output_dir, "session.json")
        with open(session_path, "w", encoding="utf-8") as session_file:
            json.dump(session_meta, session_file, indent=2)
            session_file.write("\n")

    def _on_image(self, msg: Any) -> None:
        if isinstance(msg, dict):
            self._last_img = msg
            self._last_img_ts_ns = time.time_ns()
            self._img_seq += 1

    def _on_pose(self, msg: Any) -> None:
        pose = _parse_pose(msg)
        if pose is not None:
            self._last_pose = pose
            self._t_pose = time.time()

    def _on_twist(self, msg: Any) -> None:
        twist = _parse_twist(msg)
        if twist is not None:
            self._last_twist = twist
            self._t_twist = time.time()

    def _decode_image(self, msg: Dict[str, Any]) -> Optional[Tuple[np.ndarray, str, str]]:
        try:
            height = int(msg.get("height", 0))
            width = int(msg.get("width", 0))
            encoding = str(msg.get("encoding", "bgr8")).lower()
            if height <= 0 or width <= 0:
                return None

            data = msg.get("data")
            if not isinstance(data, (bytes, bytearray)):
                if msg.get("format") == "json_b64":
                    data_b64 = msg.get("data_b64")
                    if isinstance(data_b64, str) and data_b64:
                        data = base64.b64decode(data_b64)
                elif isinstance(data, str):
                    data = base64.b64decode(data)

            if not isinstance(data, (bytes, bytearray)):
                return None

            expected = height * width * 3
            if len(data) < expected:
                return None

            img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            stored_encoding = encoding
            if self.convert_rgb and encoding == "bgr8":
                img = img[:, :, ::-1]
                stored_encoding = "rgb8"
            return img, stored_encoding, encoding
        except Exception:
            return None

    def _atomic_write_bytes(self, path: str, payload: bytes) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    def _atomic_write_npy(self, path: str, array: np.ndarray) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    def _write_image(self, path: str, img: np.ndarray, encoding: str) -> bool:
        try:
            if self.image_format == "npy":
                self._atomic_write_npy(path, img)
                return True

            if cv2 is None:
                self._atomic_write_npy(f"{path}.npy", img)
                return True

            img_to_write = img
            if encoding == "rgb8":
                img_to_write = img[:, :, ::-1]

            ext = ".jpg" if self.image_format in ("jpg", "jpeg") else ".png"
            params = []
            if self.image_format in ("jpg", "jpeg"):
                params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

            ok, encoded = cv2.imencode(ext, img_to_write, params)
            if not ok:
                return False
            self._atomic_write_bytes(path, encoded.tobytes())
            return True
        except Exception:
            return False

    def _flush_manifest(self) -> None:
        self._manifest.flush()
        os.fsync(self._manifest.fileno())

    def step(self) -> None:
        if self._last_img is None:
            return
        if self._img_seq == self._last_logged_seq:
            return

        now = time.time()
        if self.max_age_s > 0.0:
            if not self._last_pose or (now - self._t_pose) > self.max_age_s:
                return
            if not self._last_twist or (now - self._t_twist) > self.max_age_s:
                return

        decoded = self._decode_image(self._last_img)
        if decoded is None:
            return
        img, stored_encoding, source_encoding = decoded

        timestamp_ns = self._last_img_ts_ns or time.time_ns()
        sample_id = str(timestamp_ns)
        ext = "npy" if self.image_format == "npy" else ("jpg" if self.image_format in ("jpg", "jpeg") else "png")
        image_name = f"{sample_id}.{ext}"
        image_path = os.path.join(self.images_dir, image_name)
        if not self._write_image(image_path, img, stored_encoding):
            return

        vx, vy, wz = self._last_twist or (0.0, 0.0, 0.0)
        record = {
            "timestamp_ns": timestamp_ns,
            "robot_id": self.ROBOT_ID,
            "pose": self._last_pose,
            "cmd_twist": {"vx": float(vx), "vy": float(vy), "wz": float(wz)},
            "image": {
                "path": os.path.join("images", image_name),
                "height": int(img.shape[0]),
                "width": int(img.shape[1]),
                "format": ext,
                "stored_encoding": stored_encoding,
                "source_encoding": source_encoding,
            },
        }
        self._manifest.write(json.dumps(record) + "\n")
        self._record_count += 1
        if self.flush_every > 0 and self._record_count % self.flush_every == 0:
            self._flush_manifest()

        self._last_logged_seq = self._img_seq

    def cleanup(self) -> None:
        try:
            self._flush_manifest()
        except Exception:
            pass
        try:
            self._manifest.close()
        except Exception:
            pass
