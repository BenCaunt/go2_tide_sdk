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

        # Rerun setup
        try:
            rr.init(self.rerun_app_id, spawn=self.rerun_spawn)
            try:
                rr.log("world", rr.ViewCoordinates.RDF)
            except TypeError:
                pass
        except Exception:
            pass

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
