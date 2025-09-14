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
import json


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
        # Cull policy: drop cached points that lie under fresh XY voxels in the published cloud
        self.xy_cull_cached_in_fresh: bool = bool(p.get("xy_cull_cached_in_fresh", True))
        # Cached point appearance: remaining saturation [0..1] (0=gray, 1=full color)
        self.cached_saturation: float = float(p.get("cached_saturation", 0.2))

        # Online alignment (scan-to-cache) to correct drift in incoming pose
        self.align_enable: bool = bool(p.get("align_enable", True))
        self.align_period_s: float = float(p.get("align_period_s", 1.0))
        self.align_max_translation_m: float = float(p.get("align_max_translation_m", 0.3))
        self.align_max_rotation_deg: float = float(p.get("align_max_rotation_deg", 5.0))
        self.align_alpha: float = float(p.get("align_alpha", 1.0))  # 0..1 scale applied to delta
        self.align_sample_size: int = int(p.get("align_sample_size", 1200))
        self.align_use_boundary: bool = bool(p.get("align_use_boundary", True))
        self.align_boundary_neighbor_thresh: int = int(p.get("align_boundary_neighbor_thresh", 3))
        self.align_neighbor_radius_m: float = float(p.get("align_neighbor_radius_m", 0.25))
        self.align_xy_neighbor_range: int = int(p.get("align_xy_neighbor_range", 1))

        # Current alignment transform (SE2) applied to incoming scans before caching/publishing
        self._align_dx: float = 0.0
        self._align_dy: float = 0.0
        self._align_dyaw: float = 0.0  # radians
        self._last_align_time: float = 0.0

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

        # Apply current alignment transform
        pts_aligned = self._apply_se2(pts, self._align_dx, self._align_dy, self._align_dyaw)

        # Periodically estimate an incremental alignment relative to the existing cache before inserting new points
        if self.align_enable and (ts - self._last_align_time) >= self.align_period_s and len(self._cache) >= 1000:
            try:
                ddx, ddy, ddyaw = self._estimate_alignment(pts_aligned)
                if np.isfinite(ddx) and np.isfinite(ddy) and np.isfinite(ddyaw):
                    # Clamp and scale delta
                    max_t = self.align_max_translation_m
                    mag = float(np.hypot(ddx, ddy))
                    if mag > max_t and mag > 1e-6:
                        scale = max_t / mag
                        ddx *= scale
                        ddy *= scale
                    max_r = np.deg2rad(self.align_max_rotation_deg)
                    if abs(ddyaw) > max_r:
                        ddyaw = np.sign(ddyaw) * max_r
                    a = max(0.0, min(1.0, self.align_alpha))
                    ddx *= a
                    ddy *= a
                    ddyaw *= a
                    # Compose: new overall = delta ∘ current
                    self._align_dx, self._align_dy, self._align_dyaw = self._compose_se2(
                        ddx, ddy, ddyaw, self._align_dx, self._align_dy, self._align_dyaw
                    )
                    self._last_align_time = ts
                    # Publish current correction for other nodes (viewer, occupancy) to optionally use
                    try:
                        self.put(
                            "mapping/pose_correction",
                            json.dumps({
                                "dx": float(self._align_dx),
                                "dy": float(self._align_dy),
                                "dyaw": float(self._align_dyaw),
                            }),
                        )
                    except Exception:
                        pass
                    # Re-apply updated alignment to this frame before caching
                    pts_aligned = self._apply_se2(pts, self._align_dx, self._align_dy, self._align_dyaw)
            except Exception:
                pass

        # Update cache and publish
        self._update_cache(pts_aligned, ts)
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

        # Optionally cull cached points that lie under fresh XY voxels to avoid grey speckle inside the live cloud
        if self.xy_cull_cached_in_fresh and np.any(fresh_mask):
            # Quantize XY to voxel grid
            ix = np.floor(xyz[:, 0] / self.voxel).astype(np.int32)
            iy = np.floor(xyz[:, 1] / self.voxel).astype(np.int32)
            # Build set of fresh XY pairs
            fresh_xy = set(zip(ix[fresh_mask].tolist(), iy[fresh_mask].tolist()))
            if fresh_xy:
                # Build keep mask initialized to True
                keep = np.ones(xyz.shape[0], dtype=bool)
                # For cached points only, drop if their (ix,iy) is in fresh set
                for j in range(keep.shape[0]):
                    if not fresh_mask[j] and (int(ix[j]), int(iy[j])) in fresh_xy:
                        keep[j] = False
                if not keep.all():
                    xyz = xyz[keep]
                    times = times[keep]
                    fresh_mask = fresh_mask[keep]

        # Update n to match any culling that occurred
        n = xyz.shape[0]

        # Height-based colors for all points (same gradient as viewer), then desaturate cached
        # Normalize by z over the full exported set for consistency
        z_all = xyz[:, 2]
        try:
            zmin = float(np.nanmin(z_all))
            zmax = float(np.nanmax(z_all))
        except Exception:
            zmin, zmax = 0.0, 1.0
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
            tnorm_all = np.zeros_like(z_all, dtype=np.float32)
        else:
            tnorm_all = ((z_all - zmin) / (zmax - zmin)).astype(np.float32).clip(0.0, 1.0)

        colors = np.empty((n, 3), dtype=np.float32)
        # 0..1/3: blue -> cyan
        m1 = tnorm_all < 1.0 / 3.0
        if np.any(m1):
            k1 = tnorm_all[m1] * 3.0
            colors[m1, 0] = 0.0
            colors[m1, 1] = k1
            colors[m1, 2] = 1.0
        # 1/3..2/3: cyan -> yellow
        m2 = (tnorm_all >= 1.0 / 3.0) & (tnorm_all < 2.0 / 3.0)
        if np.any(m2):
            k2 = (tnorm_all[m2] - 1.0 / 3.0) * 3.0
            colors[m2, 0] = k2
            colors[m2, 1] = 1.0
            colors[m2, 2] = 1.0 - k2
        # 2/3..1: yellow -> red
        m3 = ~m1 & ~m2
        if np.any(m3):
            k3 = (tnorm_all[m3] - 2.0 / 3.0) * 3.0
            colors[m3, 0] = 1.0
            colors[m3, 1] = 1.0 - k3
            colors[m3, 2] = 0.0

        # Apply saturation: fresh keep full color, cached blend toward grayscale
        rgb = np.empty((n, 3), dtype=np.uint8)
        if np.any(fresh_mask):
            rgb[fresh_mask, :] = (colors[fresh_mask, :] * 255.0).astype(np.uint8)
        if np.any(~fresh_mask):
            sat = min(max(self.cached_saturation, 0.0), 1.0)
            cached_cols = colors[~fresh_mask, :]
            # Luminance for grayscale
            lum = (0.299 * cached_cols[:, 0] + 0.587 * cached_cols[:, 1] + 0.114 * cached_cols[:, 2]).astype(np.float32)
            gray = np.stack([lum, lum, lum], axis=1)
            mixed = sat * cached_cols + (1.0 - sat) * gray
            rgb[~fresh_mask, :] = (mixed * 255.0).astype(np.uint8)

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

    # --------------- Alignment helpers ---------------
    def _apply_se2(self, pts: np.ndarray, dx: float, dy: float, yaw: float) -> np.ndarray:
        if pts.size == 0:
            return pts
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        x = pts[:, 0]
        y = pts[:, 1]
        xr = c * x - s * y + dx
        yr = s * x + c * y + dy
        out = pts.copy()
        out[:, 0] = xr
        out[:, 1] = yr
        return out

    def _compose_se2(self, dx1: float, dy1: float, yaw1: float,
                      dx2: float, dy2: float, yaw2: float) -> Tuple[float, float, float]:
        """Return T = T1 ∘ T2 where T is applied as x' = R*yaw * x + t."""
        c = float(np.cos(yaw1))
        s = float(np.sin(yaw1))
        dx = dx1 + c * dx2 - s * dy2
        dy = dy1 + s * dx2 + c * dy2
        yaw = yaw1 + yaw2
        # Normalize yaw to [-pi, pi)
        yaw = float((yaw + np.pi) % (2 * np.pi) - np.pi)
        return dx, dy, yaw

    def _estimate_alignment(self, pts_aligned: np.ndarray) -> Tuple[float, float, float]:
        """Estimate SE2 delta that aligns current fresh points to the cached map using nearest neighbors in XY.
        Returns (ddx, ddy, ddyaw)."""
        if pts_aligned.shape[0] < 200 or not self._cache:
            return 0.0, 0.0, 0.0

        # Build XY map from cache: (ix,iy) -> list of (x,y)
        xy_map: Dict[Tuple[int, int], list] = {}
        res = self.voxel
        for (ix, iy, _iz), (x, y, _z, _t) in self._cache.items():
            key = (ix, iy)
            lst = xy_map.get(key)
            if lst is None:
                xy_map[key] = [(x, y)]
            else:
                lst.append((x, y))

        # Optionally select boundary-like points from fresh via local neighbor counts
        pts = pts_aligned
        idx_xy = np.floor(pts[:, :2] / res).astype(np.int32)
        if self.align_use_boundary:
            # Count occurrences per cell
            flat_keys = idx_xy[:, 0].astype(np.int64) << 32 | (idx_xy[:, 1].astype(np.int64) & 0xffffffff)
            # Use bincount via hash map
            counts: Dict[int, int] = {}
            for k in flat_keys.tolist():
                counts[k] = counts.get(k, 0) + 1
            neighbor_counts = np.array([counts[k] for k in flat_keys.tolist()], dtype=np.int32)
            mask = neighbor_counts <= int(self.align_boundary_neighbor_thresh)
            cand = np.where(mask)[0]
            if cand.shape[0] < 100:  # fallback to all
                cand = np.arange(pts.shape[0])
        else:
            cand = np.arange(pts.shape[0])

        # Find correspondences via XY voxel neighborhood search
        pairs_p = []  # fresh
        pairs_q = []  # cached
        rng = int(max(0, self.align_xy_neighbor_range))
        r_max2 = float(self.align_neighbor_radius_m ** 2)
        # Shuffle candidates to reduce spatial bias; then cap sample size
        if cand.shape[0] > self.align_sample_size:
            sel = np.random.choice(cand, size=self.align_sample_size, replace=False)
        else:
            sel = cand
        for i in sel:
            ix, iy = int(idx_xy[i, 0]), int(idx_xy[i, 1])
            best = None
            best_d2 = r_max2
            for dx in range(-rng, rng + 1):
                for dy in range(-rng, rng + 1):
                    lst = xy_map.get((ix + dx, iy + dy))
                    if not lst:
                        continue
                    x0, y0 = float(pts[i, 0]), float(pts[i, 1])
                    for (xc, yc) in lst:
                        d2 = (xc - x0) * (xc - x0) + (yc - y0) * (yc - y0)
                        if d2 <= best_d2:
                            best_d2 = d2
                            best = (xc, yc)
            if best is not None:
                pairs_p.append([pts[i, 0], pts[i, 1]])
                pairs_q.append([best[0], best[1]])

        if len(pairs_p) < 50:
            return 0.0, 0.0, 0.0

        P = np.asarray(pairs_p, dtype=np.float64)
        Q = np.asarray(pairs_q, dtype=np.float64)
        # Kabsch/Procrustes in 2D: find R,t minimizing ||R P + t - Q||
        p_mean = P.mean(axis=0)
        q_mean = Q.mean(axis=0)
        X = P - p_mean
        Y = Q - q_mean
        H = X.T @ Y  # 2x2
        try:
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[1, :] *= -1
                R = Vt.T @ U.T
            t = q_mean - (R @ p_mean)
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            return float(t[0]), float(t[1]), float(yaw)
        except Exception:
            return 0.0, 0.0, 0.0
