#!/usr/bin/env python3
"""
PathFollowNode: simple 2x P controller to follow a planned path.

Subscribes:
- planning/path (dict: {poses:[{x,y,yaw}]})
- state/pose3d (Pose3D)
- cmd/nav/cancel (any)

Publishes:
- cmd/twist (Twist2D)

Parameters:
- hz: float, control rate (default 20.0)
- k_dist: float, linear gain (default 0.8)
- k_yaw: float, angular gain (default 1.5)
- v_max: float, max linear speed (m/s) (default 0.5)
- w_max: float, max angular speed (rad/s) (default 1.2)
- waypoint_lookahead: int, indices ahead (default 2)
- waypoint_tol_m: float, distance to advance (default 0.15)
- goal_dist_tol_m: float (default 0.2)
- goal_yaw_tol_rad: float (default 0.35)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

from tide.core.node import BaseNode
from tide.models import Twist2D, Vector2
from tide.models.serialization import to_zenoh_value


def _yaw_from_quat(q: Dict[str, float]) -> float:
    try:
        qw = float(q.get("w", 1.0))
        qx = float(q.get("x", 0.0))
        qy = float(q.get("y", 0.0))
        qz = float(q.get("z", 0.0))
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    except Exception:
        return 0.0


def _ang_norm(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class PathFollowNode(BaseNode):
    GROUP = ""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        p = config or {}
        self.hz: float = float(p.get("update_rate", p.get("hz", 20.0)))
        self.k_dist: float = float(p.get("k_dist", 0.8))
        self.k_yaw: float = float(p.get("k_yaw", 1.5))
        self.v_max: float = float(p.get("v_max", 0.5))
        self.w_max: float = float(p.get("w_max", 1.2))
        self.lookahead: int = int(p.get("waypoint_lookahead", 2))
        self.wp_tol: float = float(p.get("waypoint_tol_m", 0.15))
        self.goal_dist_tol: float = float(p.get("goal_dist_tol_m", 0.2))
        self.goal_yaw_tol: float = float(p.get("goal_yaw_tol_rad", 0.35))
        self.twist_topic: str = str(p.get("twist_topic", "cmd/twist"))

        self._path: List[Dict[str, float]] = []
        self._wp_idx: int = 0
        self._active: bool = False

        self.subscribe("planning/path", self._on_path)
        self.subscribe("state/pose3d", self._on_pose)
        self.subscribe("cmd/nav/cancel", self._on_cancel)

        self._last_pose: Optional[Dict[str, Any]] = None

    def _on_path(self, msg: Dict[str, Any]):
        poses = msg.get("poses") if isinstance(msg, dict) else None
        if isinstance(poses, list) and len(poses) > 0:
            self._path = [
                {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "yaw": float(p.get("yaw", 0.0))}
                for p in poses
            ]
            self._wp_idx = 0
            self._active = True
        else:
            # Empty path clears
            self._path = []
            self._wp_idx = 0
            self._active = False

    def _on_pose(self, msg: Dict[str, Any]):
        self._last_pose = msg

    def _on_cancel(self, _msg: Any):
        self._path = []
        self._wp_idx = 0
        self._active = False
        self._publish_twist(0.0, 0.0)

    def _publish_twist(self, vx: float, wz: float):
        twist = Twist2D(linear=Vector2(x=vx, y=0.0), angular=wz)
        self.put(self.twist_topic, to_zenoh_value(twist))

    def step(self) -> None:
        if not self._active or self._last_pose is None or not self._path:
            return

        pos = self._last_pose.get("position") or {}
        x = float(pos.get("x", 0.0))
        y = float(pos.get("y", 0.0))
        yaw = _yaw_from_quat(self._last_pose.get("orientation") or {})

        # Choose waypoint with lookahead
        idx = min(self._wp_idx + self.lookahead, len(self._path) - 1)
        target = self._path[idx]
        tx = float(target["x"])
        ty = float(target["y"])

        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        e_yaw = _ang_norm(desired_yaw - yaw)

        # Waypoint progression
        # If close to current wp, advance
        cur_wp = self._path[self._wp_idx]
        if math.hypot(cur_wp["x"] - x, cur_wp["y"] - y) < self.wp_tol:
            if self._wp_idx < len(self._path) - 1:
                self._wp_idx += 1

        # Goal check on final waypoint
        last = self._path[-1]
        goal_dist = math.hypot(last["x"] - x, last["y"] - y)
        if goal_dist < self.goal_dist_tol and abs(e_yaw) < self.goal_yaw_tol:
            self._active = False
            self._publish_twist(0.0, 0.0)
            return

        # Controller
        v = max(0.0, min(self.v_max, self.k_dist * dist))
        w = max(-self.w_max, min(self.w_max, self.k_yaw * e_yaw))

        self._publish_twist(v, w)
