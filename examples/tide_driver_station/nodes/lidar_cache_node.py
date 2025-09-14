#!/usr/bin/env python3
"""
LidarCacheNode: subscribes to a streaming global-frame point cloud and
publishes a persistent, voxelized cache with per-point RGB colors:
 - Fresh (recently updated within TTL): white
 - Cached (older): grey

Config (params):
 - robot_id: Tide robot id (topic prefix)
 - update_rate: optional loop rate, not strictly required (event-driven)
 - input_topic: default 'sensor/lidar/points3d'
 - output_topic: default 'sensor/lidar/points3d_cached'
 - voxel_size_m: default 0.10
 - max_voxels: default 200000
 - fresh_ttl_s: default 0.5

Assumes incoming points are in a global/world frame already.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import time

from tide.core.node import BaseNode


class LidarCacheNode(BaseNode):
    GROUP = ""

    def __init__(self, *, config=None):
        super().__init__(config=config)
        p = config or {}

        # Parameters
        self.hz = float(p.get("update_rate", 20.0))
        self.input_topic = str(p.get("input_topic", "sensor/lidar/points3d"))
        self.output_topic = str(p.get("output_topic", "sensor/lidar/points3d_cached"))
        self.voxel = float(p.get("voxel_size_m", 0.10))
        self.max_voxels = int(p.get("max_voxels", 200_000))
        self.fresh_ttl_s = float(p.get("fresh_ttl_s", 0.5))

        # Voxel cache: (ix,iy,iz) -> (x,y,z,last_seen_ts)
        self._cache: Dict[Tuple[int, int, int], Tuple[float, float, float, float]] = {}

        # Subscribe to incoming points
        self.subscribe(self.input_topic, self._on_points)

    # --------------- Ingest ---------------
    def _decode_points(self, msg: Any) -> Optional[np.ndarray]:
        try:
            payload: Optional[Dict[str, Any]] = None
            if isinstance(msg, dict):
                payload = msg
            elif isinstance(msg, (bytes, bytearray)):
                # Try CBOR first
                try:
                    import cbor2  # type: ignore

                    payload = cbor2.loads(msg)
                except Exception:
                    payload = None
                # Fallback to JSON
                if payload is None:
                    try:
                        import json as _json

                        payload = _json.loads(msg)
                    except Exception:
                        payload = None
            if not isinstance(payload, dict):
                return None
            count = int(payload.get("count", 0))
            xyz = payload.get("xyz")
            if count <= 0 or not isinstance(xyz, (bytes, bytearray)):
                return None
            pts = np.frombuffer(xyz, dtype=np.float32).reshape((-1, 3))
            return pts
        except Exception:
            return None

    def _on_points(self, msg: Any):
        pts = self._decode_points(msg)
        if pts is None or pts.size == 0:
            return
        ts = time.monotonic()
        self._update_cache(pts, ts)
        xyz, rgb = self._export(ts)
        self._publish(xyz, rgb)

    # --------------- Cache ---------------
    def _update_cache(self, pts: np.ndarray, ts: float) -> None:
        try:
            idx = np.floor(pts / self.voxel).astype(np.int32)
            seen: Dict[Tuple[int, int, int], int] = {}
            for i in range(idx.shape[0]):
                key = (int(idx[i, 0]), int(idx[i, 1]), int(idx[i, 2]))
                seen[key] = i
            cache = self._cache
            for key, i in seen.items():
                x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
                cache[key] = (x, y, z, ts)
            if len(cache) > self.max_voxels:
                self._prune(len(cache) - self.max_voxels)
        except Exception:
            pass

    def _prune(self, over_by: int) -> None:
        if over_by <= 0 or not self._cache:
            return
        target_remove = max(int(0.2 * len(self._cache)), over_by)
        items = sorted(self._cache.items(), key=lambda kv: kv[1][3])
        for j in range(min(target_remove, len(items))):
            del self._cache[items[j][0]]

    def _export(self, now_ts: float) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self._cache)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        xyz = np.empty((n, 3), dtype=np.float32)
        times = np.empty((n,), dtype=np.float32)
        i = 0
        for (_k, (x, y, z, t)) in self._cache.items():
            xyz[i, 0] = x
            xyz[i, 1] = y
            xyz[i, 2] = z
            times[i] = t
            i += 1

        ttl = self.fresh_ttl_s
        fresh_mask = (now_ts - times) <= ttl

        # Default all to cached grey
        rgb = np.empty((n, 3), dtype=np.uint8)
        rgb[:] = (140, 140, 140)

        # Height-based colors for fresh points only (same gradient as viewer)
        if np.any(fresh_mask):
            z = xyz[fresh_mask, 2]
            try:
                s_min = float(np.nanmin(z))
                s_max = float(np.nanmax(z))
            except Exception:
                s_min, s_max = 0.0, 1.0
            if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
                tnorm = np.zeros_like(z, dtype=np.float32)
            else:
                tnorm = ((z - s_min) / (s_max - s_min)).astype(np.float32).clip(0.0, 1.0)

            c = np.empty((tnorm.shape[0], 3), dtype=np.float32)
            # 0..1/3: blue -> cyan
            m1 = tnorm < 1.0 / 3.0
            if np.any(m1):
                k1 = tnorm[m1] * 3.0
                c[m1, 0] = 0.0
                c[m1, 1] = k1
                c[m1, 2] = 1.0
            # 1/3..2/3: cyan -> yellow
            m2 = (tnorm >= 1.0 / 3.0) & (tnorm < 2.0 / 3.0)
            if np.any(m2):
                k2 = (tnorm[m2] - 1.0 / 3.0) * 3.0
                c[m2, 0] = k2
                c[m2, 1] = 1.0
                c[m2, 2] = 1.0 - k2
            # 2/3..1: yellow -> red
            m3 = ~m1 & ~m2
            if np.any(m3):
                k3 = (tnorm[m3] - 2.0 / 3.0) * 3.0
                c[m3, 0] = 1.0
                c[m3, 1] = 1.0 - k3
                c[m3, 2] = 0.0

            rgb[fresh_mask, :] = (c * 255.0).astype(np.uint8)

        return xyz, rgb

    # --------------- Publish ---------------
    def _publish(self, xyz: np.ndarray, rgb: np.ndarray) -> None:
        try:
            payload = {
                "count": int(xyz.shape[0]),
                "xyz": xyz.astype(np.float32, copy=False).tobytes(),
                "rgb": rgb.astype(np.uint8, copy=False).tobytes(),
            }
            # Encode as CBOR bytes; Tide SDK will decode to dict on subscribe side
            try:
                import cbor2  # type: ignore

                data = cbor2.dumps(payload)
            except Exception:
                # Fallback to JSON string with base64 for binary fields if CBOR unavailable
                import base64, json as _json

                data = _json.dumps(
                    {
                        "count": payload["count"],
                        "xyz_b64": base64.b64encode(payload["xyz"]).decode("ascii"),
                        "rgb_b64": base64.b64encode(payload["rgb"]).decode("ascii"),
                        "format": "json_b64",
                    }
                )
            self.put(self.output_topic, data)
        except Exception:
            pass

    def step(self) -> None:
        # Event-driven; no periodic work needed
        return
