#!/usr/bin/env python3
"""
TwistMuxNode: selects between teleop and navigation twist commands and outputs
the final `cmd/twist` to the robot.

Inputs:
- cmd/twist/teleop (Twist2D as dict)
- cmd/twist/nav (Twist2D as dict)

Output:
- cmd/twist (Twist2D)

Policy:
- Prefer teleop only when its magnitude exceeds `teleop_epsilon`.
- Otherwise, pass through nav if not stale.
- If both are inactive/stale, publish zero twist.

Parameters:
- hz: float, loop rate (default 30.0)
- teleop_epsilon: float, linear/angular threshold for considering teleop active (default 0.02)
- input_timeout_s: float, age in seconds after which an input is considered stale (default 0.5)
- output_topic: str, final twist topic (default "cmd/twist")
- teleop_topic: str, input topic for teleop (default "cmd/twist/teleop")
- nav_topic: str, input topic for nav (default "cmd/twist/nav")
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from tide.core.node import BaseNode
from tide.models import Twist2D, Vector2
from tide.models.serialization import to_zenoh_value


def _parse_twist(msg: Any) -> Optional[Tuple[float, float, float]]:
    """Return (vx, vy, wz) from a Twist2D-like dict; None if invalid."""
    try:
        if not isinstance(msg, dict):
            return None
        lin = msg.get("linear") or {}
        vx = float(lin.get("x", 0.0))
        vy = float(lin.get("y", 0.0))
        wz = float(msg.get("angular", msg.get("w", 0.0)))
        wz = float(wz)
        return vx, vy, wz
    except Exception:
        return None


class TwistMuxNode(BaseNode):
    GROUP = ""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        p = config or {}

        self.hz: float = float(p.get("update_rate", p.get("hz", 30.0)))
        self.teleop_epsilon: float = float(p.get("teleop_epsilon", 0.02))
        self.input_timeout_s: float = float(p.get("input_timeout_s", 0.5))
        self.output_topic: str = str(p.get("output_topic", "cmd/twist"))
        self.teleop_topic: str = str(p.get("teleop_topic", "cmd/twist/teleop"))
        self.nav_topic: str = str(p.get("nav_topic", "cmd/twist/nav"))

        self._last_teleop: Optional[Tuple[float, float, float]] = None
        self._last_nav: Optional[Tuple[float, float, float]] = None
        self._t_teleop: float = 0.0
        self._t_nav: float = 0.0

        self.subscribe(self.teleop_topic, self._on_teleop)
        self.subscribe(self.nav_topic, self._on_nav)

    def _on_teleop(self, msg: Any) -> None:
        parsed = _parse_twist(msg)
        if parsed is not None:
            self._last_teleop = parsed
            self._t_teleop = time.time()

    def _on_nav(self, msg: Any) -> None:
        parsed = _parse_twist(msg)
        if parsed is not None:
            self._last_nav = parsed
            self._t_nav = time.time()

    def _publish(self, vx: float, vy: float, wz: float) -> None:
        twist = Twist2D(linear=Vector2(x=float(vx), y=float(vy)), angular=float(wz))
        self.put(self.output_topic, to_zenoh_value(twist))

    def step(self) -> None:
        now = time.time()
        teleop = self._last_teleop
        nav = self._last_nav

        teleop_fresh = teleop is not None and (now - self._t_teleop) <= self.input_timeout_s
        nav_fresh = nav is not None and (now - self._t_nav) <= self.input_timeout_s

        # Active teleop if any component exceeds epsilon
        if teleop_fresh:
            tvx, tvy, twz = teleop  # type: ignore
            if abs(tvx) > self.teleop_epsilon or abs(tvy) > self.teleop_epsilon or abs(twz) > self.teleop_epsilon:
                self._publish(tvx, tvy, twz)
                return

        # Otherwise pass nav if available
        if nav_fresh and nav is not None:
            nvx, nvy, nwz = nav
            self._publish(nvx, nvy, nwz)
            return

        # Default to zero to keep robot stopped
        self._publish(0.0, 0.0, 0.0)

