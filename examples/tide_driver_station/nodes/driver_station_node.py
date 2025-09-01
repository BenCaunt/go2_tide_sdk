#!/usr/bin/env python3
"""
DriverStationNode: Tide node using pygame for controller input.

Publishes:
- cmd/twist (Twist2D)
- planning/target_pose2d (dict) when in targeting mode
- cmd/nav/goal (dict) on send goal
- cmd/nav/cancel (empty) on cancel

Subscribes:
- state/pose3d (Pose3D)
- sensor/imu/quat (Quaternion)

Configuration (YAML params for this node):
- robot_id: string (default: 'cash')
- update_rate / hz: float (default: 30.0)
- deadzone: float (default: 0.1)
- axes: { x: int, y: int, yaw: int } (default: {x:0, y:1, yaw:3})
- invert: { x: bool, y: bool, yaw: bool } (default: {x: False, y: True, yaw: False})
- scales: { linear: float m/s, angular: float rad/s } (default: {linear:0.6, angular:1.2})
- enable_button: Optional[int] (if set, must be held to move)
- target_mode_button: Optional[int] (toggle targeting mode)
- send_goal_button: Optional[int] (send goal when targeting)
- cancel_threshold: float (abs axis beyond cancels path, default 0.25)
"""

import os
import time
from typing import Optional, Dict, Any
import numpy as np  # kept for potential math; safe to remove if unused

try:
    import pygame
except Exception:
    pygame = None

from tide.core.node import BaseNode
from tide.models import Twist2D, Vector2
from tide.models.serialization import to_zenoh_value


class DriverStationNode(BaseNode):
    GROUP = ""  # No extra group; keys like '<ROBOT_ID>/cmd/twist'

    def __init__(self, *, config: Dict[str, Any] | None = None):
        super().__init__(config=config)

        # Parameters
        params = config or {}
        self.hz = float(params.get("update_rate", params.get("hz", 30.0)))
        self.deadzone = float(params.get("deadzone", 0.1))
        self.twist_topic: str = str(params.get("twist_topic", "cmd/twist"))
        self.axes = {
            "x": int(params.get("axes", {}).get("x", 0)),
            "y": int(params.get("axes", {}).get("y", 1)),
            "yaw": int(params.get("axes", {}).get("yaw", 3)),
        }
        inv = params.get("invert", {})
        self.invert = {
            "x": bool(inv.get("x", False)),
            "y": bool(inv.get("y", True)),
            "yaw": bool(inv.get("yaw", False)),
        }
        scl = params.get("scales", {})
        self.linear_scale = float(scl.get("linear", 0.6))
        self.angular_scale = float(scl.get("angular", 1.2))
        self.enable_button: Optional[int] = params.get("enable_button", None)
        self.target_mode_button: Optional[int] = params.get("target_mode_button", None)
        self.send_goal_button: Optional[int] = params.get("send_goal_button", None)
        self.cancel_threshold: float = float(params.get("cancel_threshold", 0.25))

        # Visualization moved to sensors node to avoid duplicate Rerun init

        # Joystick state
        self._js_ready = False
        self._joystick: Optional["pygame.joystick.Joystick"] = None
        self._last_log = 0.0
        self._target_mode = False
        self._target_xyyaw: Optional[Dict[str, float]] = None
        self._cancel_sent = False

        # Subscribe to relevant robot topics (Pose3D + Quaternion)
        # These will be prefixed with ROBOT_ID by BaseNode
        self.subscribe("state/pose3d")
        self.subscribe("sensor/imu/quat")

    # -- Pygame / Joystick helpers --
    def _ensure_pygame(self) -> None:
        if self._js_ready:
            return

        if pygame is None:
            print("pygame not installed. Install with: pip install pygame")
            return

        # Allow headless usage
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        try:
            pygame.init()
            pygame.joystick.init()
        except Exception as e:
            print(f"Failed to init pygame: {e}")
            return

        # Select first joystick
        try:
            if pygame.joystick.get_count() == 0:
                print("No joystick detected. Connect a controller.")
                return
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            self._js_ready = True
            print(f"DriverStation: Using joystick '{self._joystick.get_name()}' with {self._joystick.get_numaxes()} axes")
        except Exception as e:
            print(f"Failed to init joystick: {e}")

    def _apply_deadzone(self, v: float) -> float:
        return 0.0 if abs(v) < self.deadzone else v

    # -- Main loop --
    def step(self) -> None:
        # Ensure joystick
        self._ensure_pygame()

        vx = vy = wz = 0.0

        if self._js_ready and self._joystick is not None:
            try:
                # Pump events so joystick state updates
                pygame.event.pump()

                # Read axes
                def read_axis(name: str) -> float:
                    idx = self.axes.get(name, 0)
                    val = self._joystick.get_axis(idx) if idx < self._joystick.get_numaxes() else 0.0
                    if self.invert.get(name, False):
                        val = -val
                    return self._apply_deadzone(val)

                # Map axes to robot cmd
                # x -> strafe x (left/right), y -> forward/back, yaw -> angular z
                ax = read_axis("x")
                ay = read_axis("y")
                az = read_axis("yaw")

                # Targeting mode handling
                # Toggle targeting mode
                if self.target_mode_button is not None and self.target_mode_button < self._joystick.get_numbuttons():
                    if bool(self._joystick.get_button(self.target_mode_button)):
                        # Debounce: only toggle on new press; simple approach using cancel flag as edge guard
                        if not self._target_mode:
                            # Enter targeting mode; initialize target at current pose
                            pose = self.take("state/pose3d") or {}
                            pos = (pose.get("position") or {})
                            ori = (pose.get("orientation") or {})
                            yaw = 0.0
                            try:
                                qw = float(ori.get("w", 1.0)); qx = float(ori.get("x", 0.0)); qy = float(ori.get("y", 0.0)); qz = float(ori.get("z", 0.0))
                                yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
                            except Exception:
                                yaw = 0.0
                            self._target_xyyaw = {
                                "x": float((pos.get("x", 0.0))),
                                "y": float((pos.get("y", 0.0))),
                                "yaw": float(yaw),
                            }
                            self._target_mode = True
                else:
                    # No button configured; ensure mode off
                    self._target_mode = False

                if self._target_mode:
                    # While targeting, move target with sticks in world XY
                    # Use a modest rate per step derived from linear_scale (m/s) scaled by loop dt ~ 1/hz
                    rate = max(0.05, self.linear_scale / max(self.hz, 1.0))
                    if self._target_xyyaw is None:
                        self._target_xyyaw = {"x": 0.0, "y": 0.0, "yaw": 0.0}
                    self._target_xyyaw["x"] += float(ax * rate)
                    self._target_xyyaw["y"] += float(ay * rate)
                    self._target_xyyaw["yaw"] += float(az * 0.02)
                    # Publish target for visualization
                    self.put("planning/target_pose2d", {
                        "x": float(self._target_xyyaw["x"]),
                        "y": float(self._target_xyyaw["y"]),
                        "yaw": float(self._target_xyyaw["yaw"]),
                    })
                    # Send goal if button pressed
                    if self.send_goal_button is not None and self.send_goal_button < self._joystick.get_numbuttons():
                        if bool(self._joystick.get_button(self.send_goal_button)):
                            self.put("cmd/nav/goal", {
                                "x": float(self._target_xyyaw["x"]),
                                "y": float(self._target_xyyaw["y"]),
                                "yaw": float(self._target_xyyaw["yaw"]),
                            })
                            self._target_mode = False
                    # Do not drive robot while targeting
                    vx = vy = wz = 0.0
                else:
                    # Teleop mode: map axes to robot cmd
                    vx = ay * self.linear_scale
                    vy = ax * self.linear_scale
                    wz = az * self.angular_scale

                # Deadman/enable button
                if self.enable_button is not None:
                    btn_pressed = False
                    try:
                        if self.enable_button < self._joystick.get_numbuttons():
                            btn_pressed = bool(self._joystick.get_button(self.enable_button))
                    except Exception:
                        pass
                    if not btn_pressed:
                        vx = vy = wz = 0.0

                # Cancel path if user moves joystick beyond threshold and not targeting
                if not self._target_mode:
                    moved = max(abs(ax), abs(ay), abs(az)) > self.cancel_threshold
                    if moved and not self._cancel_sent:
                        self.put("cmd/nav/cancel", {"reason": "joystick_override"})
                        self._cancel_sent = True
                    elif not moved:
                        self._cancel_sent = False

            except Exception as e:
                print(f"Joystick read error: {e}")
                vx = vy = wz = 0.0

        # Publish cmd/twist
        twist = Twist2D(linear=Vector2(x=vx, y=vy), angular=wz)
        self.put(self.twist_topic, to_zenoh_value(twist))

        # Fetch latest pose + quat for periodic console logging
        pose = self.take("state/pose3d")
        quat = self.take("sensor/imu/quat")

        # Log at ~1 Hz
        now = time.time()
        if now - self._last_log > 1.0:
            self._last_log = now
            pose_txt = "pose:N/A"
            if isinstance(pose, dict):
                p = pose.get("position") or {}
                pose_txt = f"pos=({p.get('x',0):.2f},{p.get('y',0):.2f},{p.get('z',0):.2f})"
            quat_txt = "quat:N/A"
            if isinstance(quat, dict):
                quat_txt = f"quat=({quat.get('w',1):.2f},{quat.get('x',0):.2f},{quat.get('y',0):.2f},{quat.get('z',0):.2f})"
            print(f"DriverStation cmd: vx={vx:.2f} vy={vy:.2f} wz={wz:.2f} | {pose_txt} {quat_txt}")
