#!/usr/bin/env python3
"""
Go2SensorsNode (viewer): subscribes to Tide camera + LiDAR topics that the
go2_tide_bridge publishes, and visualizes them with Rerun. No direct Go2
connection from this node (single client constraint).
"""

from typing import Any, Dict, Optional
import numpy as np

from tide.core.node import BaseNode
import rerun as rr


class Go2SensorsNode(BaseNode):
    GROUP = ""

    def __init__(self, *, config=None):
        super().__init__(config=config)
        p = config or {}

        # Update rate (viewer loop)
        self.hz = float(p.get("update_rate", 20.0))

        # Rerun
        self.rerun_spawn = bool(p.get("rerun_spawn", True))
        self.rerun_app_id = p.get("rerun_app_id", f"go2/{self.ROBOT_ID}")

        # Latest received messages
        self._last_img: Optional[Dict[str, Any]] = None
        self._last_pc: Optional[Dict[str, Any]] = None

        # Point cloud coloring
        # Modes: 'auto' (use RGB if provided, else fallback),
        #        'height' (by z), 'distance' (by Euclidean distance),
        #        'rgb' (only use provided RGB), 'none' (no colors)
        self.pc_color_mode = str(p.get("pc_color_mode", "height")).lower()

        # Rerun setup
        rr.init(self.rerun_app_id, spawn=self.rerun_spawn)
        rr.log("world", rr.ViewCoordinates.RDF)
        # Subscribe to topics from go2_tide_bridge
        self.subscribe("sensor/camera/front/image", self._on_image)
        self.subscribe("sensor/lidar/points3d", self._on_points)

    def _on_image(self, msg: Dict[str, Any]):
        self._last_img = msg

    def _on_points(self, msg: Dict[str, Any]):
        self._last_pc = msg

    # ----------------- Publish/visualize -----------------
    def _render_image(self, img_msg: Dict[str, Any]):
        try:
            h = int(img_msg.get("height", 0))
            w = int(img_msg.get("width", 0))
            enc = img_msg.get("encoding", "bgr8").lower()
            data = img_msg.get("data")
            if not h or not w or not isinstance(data, (bytes, bytearray)):
                return
            img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
            rgb = img[:, :, ::-1] if enc == "bgr8" else img
            rr.log("/camera/front", rr.Image(rgb))
        except Exception:
            pass

    def _render_pointcloud(self, pc_msg: Dict[str, Any]):
        try:
            count = int(pc_msg.get("count", 0))
            xyz = pc_msg.get("xyz")
            if count <= 0 or not isinstance(xyz, (bytes, bytearray)):
                return
            pts = np.frombuffer(xyz, dtype=np.float32).reshape((-1, 3))

            colors = None

            # Try to use provided RGB if available and mode allows
            rgb_bytes = pc_msg.get("rgb")
            if (
                self.pc_color_mode in ("auto", "rgb")
                and isinstance(rgb_bytes, (bytes, bytearray))
                and len(rgb_bytes) >= count * 3
            ):
                colors = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((-1, 3))

            # If no colors yet, compute by height or distance depending on mode
            if colors is None and self.pc_color_mode in ("auto", "height", "distance"):
                if self.pc_color_mode in ("auto", "height"):
                    scalars = pts[:, 2]  # z height
                else:
                    scalars = np.linalg.norm(pts, axis=1)  # distance

                # Normalize scalars to [0,1] with simple min-max; handle degenerate case
                s_min = float(np.nanmin(scalars))
                s_max = float(np.nanmax(scalars))
                if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
                    norm = np.zeros_like(scalars, dtype=np.float32)
                else:
                    norm = ((scalars - s_min) / (s_max - s_min)).astype(np.float32)

                # Map to a simple blue->cyan->yellow->red gradient
                # This avoids external colormap deps and works with uint8 RGB
                t = norm.clip(0.0, 1.0)
                # Piecewise: 0..0.33 blue->cyan, 0.33..0.66 cyan->yellow, 0.66..1 yellow->red
                c = np.empty((t.shape[0], 3), dtype=np.float32)
                # segment 1
                m1 = t < 1/3
                k1 = np.zeros_like(t)
                k1[m1] = t[m1] * 3.0
                c[m1, 0] = 0.0
                c[m1, 1] = k1[m1]
                c[m1, 2] = 1.0
                # segment 2
                m2 = (t >= 1/3) & (t < 2/3)
                k2 = np.zeros_like(t)
                k2[m2] = (t[m2] - 1/3) * 3.0
                c[m2, 0] = k2[m2]
                c[m2, 1] = 1.0
                c[m2, 2] = 1.0 - k2[m2]
                # segment 3
                m3 = t >= 2/3
                k3 = np.zeros_like(t)
                k3[m3] = (t[m3] - 2/3) * 3.0
                c[m3, 0] = 1.0
                c[m3, 1] = 1.0 - k3[m3]
                c[m3, 2] = 0.0

                colors = (c * 255.0).astype(np.uint8)

            # If still no colors (mode 'none' or failed), log points without colors
            if colors is not None:
                rr.log("/lidar/points", rr.Points3D(pts, colors=colors))
            else:
                rr.log("/lidar/points", rr.Points3D(pts))
        except Exception:
            pass

    def step(self) -> None:
        if self._last_img is not None:
            self._render_image(self._last_img)
            self._last_img = None
        if self._last_pc is not None:
            self._render_pointcloud(self._last_pc)
            self._last_pc = None
