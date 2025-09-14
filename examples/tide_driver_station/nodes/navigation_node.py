#!/usr/bin/env python3
"""
NavigationNode: SE(2) A* planner over OccupancyGrid2D with 16 heading bins.

Subscribes:
- mapping/occupancy (OccupancyGrid2D as dict)
- state/pose3d (Pose3D)
- cmd/nav/goal (dict: {x,y,yaw})
- cmd/nav/cancel (any)

Publishes:
- planning/path (dict: {poses:[{x,y,yaw}], frame:"map"})
- planning/target_pose2d (echo of current target for UI)

Parameters:
- hz: float, planner loop rate (default 5.0)
- heading_bins: int, number of orientation bins (default 16)
- step_cells: int, forward step in cells per primitive (default 2)
- turn_bins: int, allowed bin delta per step (default 1)
- robot_radius_m: float, collision radius in meters (default 0.28)
- allow_unknown: bool, treat unknown cells (-1) as free (default False)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import heapq
import math
import numpy as np
import time

from tide.core.node import BaseNode


def angle_normalize(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class NavigationNode(BaseNode):
    GROUP = ""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        p = config or {}
        self.hz: float = float(p.get("update_rate", p.get("hz", 5.0)))
        self.heading_bins: int = int(p.get("heading_bins", 16))
        self.step_cells: int = int(p.get("step_cells", 2))
        self.turn_bins: int = int(p.get("turn_bins", 1))
        self.robot_radius_m: float = float(p.get("robot_radius_m", 0.28))
        self.allow_unknown: bool = bool(p.get("allow_unknown", False))
        # Turning behavior tuning
        self.allow_turn_in_place: bool = bool(p.get("allow_turn_in_place", True))
        self.cost_turn_in_place: float = float(p.get("cost_turn_in_place", 0.4))
        self.cost_turn_while_moving: float = float(p.get("cost_turn_while_moving", 0.8))
        # Replan hysteresis
        self.min_replan_period_s: float = float(p.get("min_replan_period_s", 0.3))

        # Inputs
        self._grid_msg: Optional[Dict[str, Any]] = None
        self._grid_arr: Optional[np.ndarray] = None
        self._pose: Optional[Dict[str, Any]] = None
        self._goal: Optional[Dict[str, float]] = None
        self._path_world: Optional[List[Tuple[float, float, float]]] = None

        # Cache for replanning trigger
        self._last_grid_version: int = 0
        self._grid_version: int = 0
        self._t_last_plan: float = 0.0

        self.subscribe("mapping/occupancy", self._on_grid)
        self.subscribe("state/pose3d", self._on_pose)
        self.subscribe("cmd/nav/goal", self._on_goal)
        self.subscribe("cmd/nav/cancel", self._on_cancel)

    # ---- Callbacks ----
    def _on_grid(self, msg: Dict[str, Any]):
        try:
            w = int(msg.get("width", 0))
            h = int(msg.get("height", 0))
            data = msg.get("data")
            if w > 0 and h > 0 and isinstance(data, list) and len(data) >= w * h:
                arr = np.array(data[: w * h], dtype=np.int16).reshape((h, w))
                self._grid_arr = arr
                self._grid_msg = msg
                self._grid_version += 1
        except Exception:
            pass

    def _on_pose(self, msg: Dict[str, Any]):
        self._pose = msg

    def _on_goal(self, msg: Any):
        # Accept dict with x,y(,yaw) in world meters
        try:
            if isinstance(msg, dict):
                x = float(msg.get("x"))
                y = float(msg.get("y"))
                yaw = float(msg.get("yaw", 0.0))
                self._goal = {"x": x, "y": y, "yaw": yaw}
                # Echo for UI
                self.put("planning/target_pose2d", msg)
        except Exception:
            pass

    def _on_cancel(self, _msg: Any):
        self._goal = None
        self._path_world = None
        # Publish empty path to clear followers/viewers
        self.put("planning/path", {"poses": [], "frame": "map"})

    # ---- Utilities ----
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        g = self._grid_msg or {}
        res = float(g.get("resolution", 0.1))
        ox = float(g.get("origin_x", 0.0))
        oy = float(g.get("origin_y", 0.0))
        ix = int(math.floor((x - ox) / res))
        iy = int(math.floor((y - oy) / res))
        return ix, iy

    def _grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        g = self._grid_msg or {}
        res = float(g.get("resolution", 0.1))
        ox = float(g.get("origin_x", 0.0))
        oy = float(g.get("origin_y", 0.0))
        x = ox + (ix + 0.5) * res
        y = oy + (iy + 0.5) * res
        return x, y

    def _collision_at(self, ix: int, iy: int) -> bool:
        arr = self._grid_arr
        g = self._grid_msg or {}
        if arr is None:
            return True
        h, w = arr.shape
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return True
        res = float(g.get("resolution", 0.1))
        rad_cells = max(1, int(math.ceil(self.robot_radius_m / res)))
        x0 = max(0, ix - rad_cells)
        x1 = min(w - 1, ix + rad_cells)
        y0 = max(0, iy - rad_cells)
        y1 = min(h - 1, iy + rad_cells)
        # Check disk neighborhood
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if (xx - ix) * (xx - ix) + (yy - iy) * (yy - iy) <= rad_cells * rad_cells:
                    v = int(arr[yy, xx])
                    if v >= 100:
                        return True
                    if v < 0 and not self.allow_unknown:
                        return True
        return False

    def _line_collision(self, ix0: int, iy0: int, ix1: int, iy1: int) -> bool:
        # Sample Bresenham-style between two cells and check footprint
        dx = ix1 - ix0
        dy = iy1 - iy0
        steps = max(abs(dx), abs(dy), 1)
        for s in range(steps + 1):
            ix = ix0 + int(round(dx * s / steps))
            iy = iy0 + int(round(dy * s / steps))
            if self._collision_at(ix, iy):
                return True
        return False

    # ---- Planning ----
    def _plan(self, start_xyyaw: Tuple[float, float, float], goal_xyyaw: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        if self._grid_arr is None or self._grid_msg is None:
            return None
        arr = self._grid_arr
        h, w = arr.shape
        bins = self.heading_bins
        step_cells = max(1, int(self.step_cells))
        turn_bins = max(0, int(self.turn_bins))

        sx, sy, syaw = start_xyyaw
        gx, gy, gyaw = goal_xyyaw
        six, siy = self._world_to_grid(sx, sy)
        gix, giy = self._world_to_grid(gx, gy)
        if six < 0 or siy < 0 or six >= w or siy >= h:
            return None
        if gix < 0 or giy < 0 or gix >= w or giy >= h:
            return None
        if self._collision_at(six, siy) or self._collision_at(gix, giy):
            return None

        def yaw_to_bin(yaw: float) -> int:
            y = (yaw + 2 * math.pi) % (2 * math.pi)
            return int(round((y / (2 * math.pi)) * bins)) % bins

        def bin_to_yaw(ih: int) -> float:
            return (2 * math.pi * ih) / bins

        start_h = yaw_to_bin(syaw)
        goal_h = yaw_to_bin(gyaw)

        # A* structures
        start = (six, siy, start_h)
        goal = (gix, giy, goal_h)
        open_heap: List[Tuple[float, Tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (0.0, start))
        gscore: Dict[Tuple[int, int, int], float] = {start: 0.0}
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

        def heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
            ax, ay, ah = a
            bx, by, bh = b
            dx = ax - bx
            dy = ay - by
            d = math.hypot(dx, dy)
            dh = min(abs(ah - bh), bins - abs(ah - bh)) / bins
            return d + 0.5 * dh

        # Goal check when close enough in pos, heading tolerance relaxed
        def is_goal(state: Tuple[int, int, int]) -> bool:
            x, y, hbin = state
            if abs(x - gix) + abs(y - giy) <= 1:
                return True
            return False

        visited = set()
        while open_heap:
            _f, cur = heapq.heappop(open_heap)
            if cur in visited:
                continue
            visited.add(cur)
            if is_goal(cur):
                # Reconstruct path (grid -> world)
                path_cells: List[Tuple[int, int, int]] = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path_cells.append(cur)
                path_cells.reverse()
                path_world: List[Tuple[float, float, float]] = []
                for (cx, cy, ch) in path_cells:
                    wx, wy = self._grid_to_world(cx, cy)
                    wyaw = bin_to_yaw(ch)
                    path_world.append((wx, wy, wyaw))
                return path_world

            cx, cy, ch = cur
            # Generate neighbors: turn {-turn_bins..+turn_bins} and move forward step_cells
            for db in range(-turn_bins, turn_bins + 1):
                nh = (ch + db) % bins
                # 1) Move forward primitive with heading change db
                yaw = bin_to_yaw(nh)
                dx = math.cos(yaw) * step_cells
                dy = math.sin(yaw) * step_cells
                nx = int(round(cx + dx))
                ny = int(round(cy + dy))
                if 0 <= nx < w and 0 <= ny < h:
                    # Collision along the segment
                    if not self._line_collision(cx, cy, nx, ny):
                        ns = (nx, ny, nh)
                        move_cost = float(step_cells)
                        turn_cost = self.cost_turn_while_moving * float(abs(db))
                        cost = gscore[cur] + move_cost + turn_cost
                        if cost < gscore.get(ns, float("inf")):
                            gscore[ns] = cost
                            f = cost + heuristic(ns, goal)
                            parent[ns] = cur
                            heapq.heappush(open_heap, (f, ns))

                # 2) Optional in-place turn primitive (no translation)
                if self.allow_turn_in_place and db != 0:
                    # Remain in the same cell; ensure the footprint is valid at current cell
                    if not self._collision_at(cx, cy):
                        ns_turn = (cx, cy, nh)
                        turn_only_cost = gscore[cur] + self.cost_turn_in_place * float(abs(db))
                        if turn_only_cost < gscore.get(ns_turn, float("inf")):
                            gscore[ns_turn] = turn_only_cost
                            f2 = turn_only_cost + heuristic(ns_turn, goal)
                            parent[ns_turn] = cur
                            heapq.heappush(open_heap, (f2, ns_turn))

        return None

    def _current_pose_xyyaw(self) -> Optional[Tuple[float, float, float]]:
        p = self._pose or {}
        pos = p.get("position") or {}
        try:
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            ori = p.get("orientation") or {}
            qw = float(ori.get("w", 1.0))
            qx = float(ori.get("x", 0.0))
            qy = float(ori.get("y", 0.0))
            qz = float(ori.get("z", 0.0))
            # yaw from quaternion
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            return (x, y, yaw)
        except Exception:
            return None

    # ---- Main loop ----
    def step(self) -> None:
        # If goal present and we have grid + pose, ensure a path exists (or replan on grid change)
        goal = self._goal
        start = self._current_pose_xyyaw()
        if goal is None or start is None or self._grid_arr is None:
            return

        # Replan if no path or grid updated or path invalidated
        need_replan = False
        if self._path_world is None:
            need_replan = True
        elif self._last_grid_version != self._grid_version:
            need_replan = True
        else:
            # Check remaining waypoints for collision
            for (wx, wy, _wyaw) in self._path_world:
                ix, iy = self._world_to_grid(wx, wy)
                if self._collision_at(ix, iy):
                    need_replan = True
                    break

        now = time.time()
        if need_replan and (self._path_world is None or (now - self._t_last_plan) >= self.min_replan_period_s):
            start_xyyaw = start
            goal_yaw_val = float(goal.get("yaw", 0.0))
            goal_xyyaw = (float(goal["x"]), float(goal["y"]), goal_yaw_val)
            path = self._plan(start_xyyaw, goal_xyyaw)
            self._last_grid_version = self._grid_version
            self._path_world = path
            self._t_last_plan = now
            # Publish path (or empty if None)
            poses = []
            if path is not None:
                for (x, y, yaw) in path:
                    poses.append({"x": float(x), "y": float(y), "yaw": float(yaw)})
                # Ensure final pose yaw matches requested goal yaw to aid final alignment
                if len(poses) > 0:
                    poses[-1]["yaw"] = float(goal_yaw_val)
            self.put("planning/path", {"poses": poses, "frame": "map"})
