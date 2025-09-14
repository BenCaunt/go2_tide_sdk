#!/usr/bin/env python3
"""
OccupancyGridNode: subscribes to LiDAR points and produces a cached
2D occupancy grid suitable for planning and visualization.

Publishes:
- mapping/occupancy (OccupancyGrid2D)
- mapping/occupancy/image (JSON with base64 RGB image)

Rules:
- Mark a cell occupied if it has at least `occ_min_points` points with
  height in [min_z, max_z] (defaults: [0.2m, 1.0m], 5 points).
- Mark a cell free if it has near-ground points (z < free_z_max) and is
  not occupied in the current frame.
- Maintain a persistent cache; cells not observed in a given frame keep
  their previous values and do not "decay" to unknown.

Notes:
- Assumes incoming point cloud is already in a consistent/global frame.
- Origin is initialized on first received pose to center robot in the grid.
  If no pose is ever received, origin defaults to a symmetric grid around (0,0).
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from tide.core.node import BaseNode
from tide.models.serialization import to_zenoh_value

from tide.core.utils import add_project_root_to_path

add_project_root_to_path(__file__)

from occupancy_grid2d import OccupancyGrid2D


class OccupancyGridNode(BaseNode):
    GROUP = ""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        p = config or {}

        # Update rate
        self.hz: float = float(p.get("update_rate", p.get("hz", 10.0)))

        # Grid parameters
        self.resolution: float = float(p.get("resolution", 0.1))  # meters per cell
        self.width: int = int(p.get("width", 400))  # 40m @ 0.1m
        self.height: int = int(p.get("height", 400))

        # Occupancy filter params
        self.min_z: float = float(p.get("min_z", 0.2))
        self.max_z: float = float(p.get("max_z", 1.0))
        self.free_z_max: float = float(p.get("free_z_max", 0.1))
        self.occ_min_points: int = int(p.get("occ_min_points", 5))
        self.free_min_points: int = int(p.get("free_min_points", 1))
        # Clearing policy
        self.clear_occupied: bool = bool(p.get("clear_occupied", True))
        self.clear_free_min_points: int = int(p.get("clear_free_min_points", max(1, self.free_min_points)))
        # Free-space carving via raycasting from robot pose to each point
        self.raycast_free: bool = bool(p.get("raycast_free", True))
        self.raycast_stride: int = max(1, int(p.get("raycast_stride", 3)))

        # Point cloud frame: if True, interpret points in robot/base frame and transform by latest pose.
        # If False (default), interpret points as already in world frame.
        self.pc_in_robot_frame: bool = bool(p.get("pc_in_robot_frame", False))

        # Grid origin (world meters). Initialized on first pose; else defaults around (0,0)
        self.origin_x: Optional[float] = None
        self.origin_y: Optional[float] = None

        # Persistent occupancy data: -1 unknown, 0 free, >=100 occupied
        self._grid: np.ndarray = np.full((self.height, self.width), -1, dtype=np.int16)

        # Scratch buffers reused across frames
        self._obs_counts: np.ndarray = np.zeros((self.height, self.width), dtype=np.int16)
        self._free_counts: np.ndarray = np.zeros((self.height, self.width), dtype=np.int16)

        # Latest inputs
        self._last_pc: Optional[Dict[str, Any]] = None
        self._last_pose: Optional[Dict[str, Any]] = None
        self._last_path: Optional[List[Dict[str, float]]] = None

        # Subscribe to inputs
        self.points_topic: str = str(p.get("points_topic", "sensor/lidar/points3d"))
        self.subscribe(self.points_topic, self._on_points)
        self.subscribe("state/pose3d", self._on_pose)
        # Optional: subscribe to planned path for overlay in occupancy image
        self.subscribe("planning/path", self._on_path)

    # -- Callbacks --
    def _on_points(self, msg: Dict[str, Any]):
        self._last_pc = msg

    def _on_pose(self, msg: Dict[str, Any]):
        self._last_pose = msg

        # Initialize origin on first pose to keep robot centered
        if self.origin_x is None or self.origin_y is None:
            try:
                pos = msg.get("position") or {}
                px = float(pos.get("x", 0.0))
                py = float(pos.get("y", 0.0))
                span_x = self.width * self.resolution
                span_y = self.height * self.resolution
                self.origin_x = px - 0.5 * span_x
                self.origin_y = py - 0.5 * span_y
            except Exception:
                pass

    def _on_path(self, msg: Dict[str, Any]):
        try:
            poses = msg.get("poses") if isinstance(msg, dict) else None
            if isinstance(poses, list) and len(poses) > 0:
                self._last_path = [
                    {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0))}
                    for p in poses
                ]
            else:
                self._last_path = None
        except Exception:
            self._last_path = None

    # -- Utilities --
    def _ensure_origin(self) -> Tuple[float, float]:
        if self.origin_x is None or self.origin_y is None:
            # Default origin to symmetric grid around (0,0)
            span_x = self.width * self.resolution
            span_y = self.height * self.resolution
            self.origin_x = -0.5 * span_x
            self.origin_y = -0.5 * span_y
        return float(self.origin_x), float(self.origin_y)

    def _decode_points(self, pc_msg: Dict[str, Any]) -> Optional[np.ndarray]:
        """Decode various supported LiDAR point formats into Nx3 float32 array."""
        try:
            count = int(pc_msg.get("count", 0))
            xyz = pc_msg.get("xyz")
            if count > 0:
                if isinstance(xyz, (bytes, bytearray)):
                    pts = np.frombuffer(xyz, dtype=np.float32).reshape((-1, 3))
                    return pts
                # JSON fallback with base64
                if pc_msg.get("format") == "json_b64":
                    try:
                        import base64

                        b64 = pc_msg.get("xyz_b64")
                        if isinstance(b64, str):
                            raw = base64.b64decode(b64)
                            pts = np.frombuffer(raw, dtype=np.float32).reshape((-1, 3))
                            return pts
                    except Exception:
                        pass
        except Exception:
            pass

        # Fallback: explicit list of points
        try:
            pts_list = pc_msg.get("points") or pc_msg.get("xyz")
            if isinstance(pts_list, list) and len(pts_list) > 0:
                pts = np.asarray(pts_list, dtype=np.float32)
                if pts.ndim == 2 and pts.shape[1] >= 3:
                    return pts[:, :3]
        except Exception:
            pass

        return None

    def _points_to_indices(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map Nx3 points to grid indices; returns (iy, ix, mask_inside)."""
        ox, oy = self._ensure_origin()
        # Compute indices (floored)
        ix = np.floor((pts[:, 0] - ox) / self.resolution).astype(np.int32)
        iy = np.floor((pts[:, 1] - oy) / self.resolution).astype(np.int32)
        inside = (ix >= 0) & (iy >= 0) & (ix < self.width) & (iy < self.height)
        return iy, ix, inside

    def _accumulate_counts(self, iy: np.ndarray, ix: np.ndarray, mask: np.ndarray, into: np.ndarray) -> None:
        # Zero buffer and accumulate counts at (iy, ix) for mask==True
        into.fill(0)
        if not np.any(mask):
            return
        sel_iy = iy[mask]
        sel_ix = ix[mask]
        # Use flat indices to add-at
        flat = sel_iy.astype(np.int64) * self.width + sel_ix.astype(np.int64)
        np.add.at(into.ravel(), flat, 1)

    def _publish_grid(self):
        # Convert to Tide OccupancyGrid2D and publish
        msg = OccupancyGrid2D(
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            origin_x=float(self.origin_x if self.origin_x is not None else 0.0),
            origin_y=float(self.origin_y if self.origin_y is not None else 0.0),
            data=self._grid.ravel().astype(np.int16).tolist(),
        )
        self.put("mapping/occupancy", to_zenoh_value(msg))

        # Also publish an RGB image for quick viewing (unknown gray, free green, occupied red)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        arr = self._grid
        unknown = arr < 0
        free = arr == 0
        occ = arr >= 100
        img[unknown] = (80, 80, 80)
        img[free] = (0, 255, 0)
        img[occ] = (255, 0, 0)

        # Optional: overlay robot position/orientation directly into the image
        try:
            pose = self._last_pose or {}
            pos = pose.get("position") or {}
            px = float(pos.get("x", 0.0))
            py = float(pos.get("y", 0.0))
            ox, oy = self._ensure_origin()
            if self.resolution > 0.0:
                ix = int(np.round((px - ox) / self.resolution))
                iy = int(np.round((py - oy) / self.resolution))
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    # Draw a small black cross at the robot position
                    for dx in range(-2, 3):
                        x = ix + dx
                        if 0 <= x < self.width:
                            img[iy, x] = (0, 0, 0)
                    for dy in range(-2, 3):
                        y = iy + dy
                        if 0 <= y < self.height:
                            img[y, ix] = (0, 0, 0)
                    # Heading arrow (~0.5 m)
                    yaw = 0.0
                    try:
                        ori = pose.get("orientation") or {}
                        qw = float(ori.get("w"))
                        qx = float(ori.get("x"))
                        qy = float(ori.get("y"))
                        qz = float(ori.get("z"))
                        yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
                    except Exception:
                        yaw = 0.0
                    L_cells = int(max(1, np.round(0.5 / self.resolution)))
                    dx = np.cos(yaw)
                    dy = np.sin(yaw)
                    for t in range(1, L_cells + 1):
                        x = int(np.round(ix + dx * t))
                        y = int(np.round(iy + dy * t))
                        if 0 <= x < self.width and 0 <= y < self.height:
                            img[y, x] = (0, 0, 0)
                        else:
                            break
        except Exception:
            pass

        # Optional: overlay planned path on the occupancy image (blue polyline)
        try:
            if self._last_path and len(self._last_path) > 1:
                ox, oy = self._ensure_origin()
                col = (0, 128, 255)  # BGR: light orange/blue-ish for contrast

                def to_ixy(wx: float, wy: float) -> Optional[Tuple[int, int]]:
                    if self.resolution <= 0:
                        return None
                    ix = int(np.round((wx - ox) / self.resolution))
                    iy = int(np.round((wy - oy) / self.resolution))
                    if 0 <= ix < self.width and 0 <= iy < self.height:
                        return ix, iy
                    return None

                def draw_line(ix0: int, iy0: int, ix1: int, iy1: int) -> None:
                    dx = abs(ix1 - ix0)
                    dy = -abs(iy1 - iy0)
                    sx = 1 if ix0 < ix1 else -1
                    sy = 1 if iy0 < iy1 else -1
                    err = dx + dy
                    x, y = ix0, iy0
                    while True:
                        if 0 <= x < self.width and 0 <= y < self.height:
                            img[y, x] = col
                        if x == ix1 and y == iy1:
                            break
                        e2 = 2 * err
                        if e2 >= dy:
                            err += dy
                            x += sx
                        if e2 <= dx:
                            err += dx
                            y += sy

                pts = [to_ixy(p["x"], p["y"]) for p in self._last_path]
                prev = None
                for pt in pts:
                    if pt is not None and prev is not None:
                        draw_line(prev[0], prev[1], pt[0], pt[1])
                    prev = pt
        except Exception:
            pass

        payload = {
            "height": int(self.height),
            "width": int(self.width),
            "encoding": "rgb8",
            "format": "json_b64",
            "data_b64": base64.b64encode(img.tobytes()).decode("ascii"),
        }
        # Publish as JSON string for compatibility
        self.put("mapping/occupancy/image", json.dumps(payload))

    def _raycast_free(self, ix0: int, iy0: int, ix1: int, iy1: int) -> None:
        """Mark free cells along the line from (ix0,iy0) to (ix1,iy1), excluding the endpoint."""
        dx = abs(ix1 - ix0)
        dy = -abs(iy1 - iy0)
        sx = 1 if ix0 < ix1 else -1
        sy = 1 if iy0 < iy1 else -1
        err = dx + dy
        x, y = ix0, iy0
        while True:
            if x == ix1 and y == iy1:
                break
            if 0 <= x < self.width and 0 <= y < self.height:
                # Clear to free; if clear_occupied is false, keep occupied cells
                if self.clear_occupied or self._grid[y, x] < 100:
                    self._grid[y, x] = 0
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    # -- Main loop --
    def step(self) -> None:
        pc_msg = self._last_pc
        if not isinstance(pc_msg, dict):
            return

        pts = self._decode_points(pc_msg)
        if pts is None or pts.shape[0] == 0:
            return

        # Optional: transform from robot/base frame to world using latest pose
        if self.pc_in_robot_frame:
            pose = self._last_pose
            if not isinstance(pose, dict):
                # Without a pose we cannot place the points into the world; skip this frame
                return
            try:
                pos = pose.get("position") or {}
                px = float(pos.get("x", 0.0))
                py = float(pos.get("y", 0.0))
                ori = pose.get("orientation") or {}
                qw = float(ori.get("w", 1.0))
                qx = float(ori.get("x", 0.0))
                qy = float(ori.get("y", 0.0))
                qz = float(ori.get("z", 0.0))
                # yaw from quaternion
                yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
                c, s = float(np.cos(yaw)), float(np.sin(yaw))
                # Rotate XY then translate
                x_local = pts[:, 0]
                y_local = pts[:, 1]
                xw = c * x_local - s * y_local + px
                yw = s * x_local + c * y_local + py
                pts = np.stack([xw, yw, pts[:, 2]], axis=1)
            except Exception:
                return

        # Keep previous occupancy (cache). Only update cells observed in this frame.
        iy, ix, inside = self._points_to_indices(pts)
        if not np.any(inside):
            # Nothing falls inside grid; still publish current cache occasionally
            self._publish_grid()
            self._last_pc = None
            return

        pts_in = pts[inside]
        iy = iy[inside]
        ix = ix[inside]

        # Masks by height
        z = pts_in[:, 2]
        obs_mask = (z >= self.min_z) & (z <= self.max_z)
        free_mask = (z <= self.free_z_max)

        # Accumulate counts this frame
        self._accumulate_counts(iy, ix, obs_mask, self._obs_counts)
        self._accumulate_counts(iy, ix, free_mask, self._free_counts)

        # Update occupied cells (z within obstacle band) with per-cell count threshold
        occ_cells = self._obs_counts >= int(self.occ_min_points)

        # Free-space carving by raycasting from robot to each endpoint
        if self.raycast_free and isinstance(self._last_pose, dict):
            try:
                pos = self._last_pose.get("position") or {}
                px = float(pos.get("x", 0.0))
                py = float(pos.get("y", 0.0))
                ox, oy = self._ensure_origin()
                ix0 = int(np.round((px - ox) / self.resolution))
                iy0 = int(np.round((py - oy) / self.resolution))
                # Downsample endpoints to reduce workload
                for j in range(0, ix.shape[0], self.raycast_stride):
                    x1 = int(ix[j])
                    y1 = int(iy[j])
                    self._raycast_free(ix0, iy0, x1, y1)
            except Exception:
                pass

        # Apply occupied markings after carving free so endpoints win
        if np.any(occ_cells):
            self._grid[occ_cells] = 100

        # Additionally, update free cells where we saw ground points and not newly marked occupied
        free_cells = self._free_counts >= int(self.clear_free_min_points)
        if np.any(free_cells):
            if self.clear_occupied:
                # Clear regardless of previous value when we have sufficient free evidence and not newly occupied in this frame
                mask = free_cells & (~occ_cells)
                self._grid[mask] = 0
            else:
                # Only clear unknown/free cells
                mask = free_cells & (~occ_cells) & (self._grid < 100)
                self._grid[mask] = 0

        # Publish results this step
        self._publish_grid()
        self._last_pc = None
