#!/usr/bin/env python3
"""
Go2SensorsNode (viewer): subscribes to Tide camera + LiDAR topics that the
go2_tide_bridge publishes, and visualizes them with Rerun. No direct Go2
connection from this node (single client constraint).
"""

from typing import Any, Dict, Optional
import numpy as np
import json
import base64
import math
import sys
import os

from tide.core.node import BaseNode
import rerun as rr

# Add parent directory to path for camera_lidar_fusion import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from camera_lidar_fusion import CameraLidarFusion, create_fusion
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False


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
        self._last_pose: Optional[Dict[str, Any]] = None
        self._pose_cache: Optional[Dict[str, Any]] = None

        # Robot dimensions in meters (standing): 0.70 x 0.31 x 0.40
        # Used for a simple 3D box visualization in Rerun.
        self._robot_half_sizes = np.array([0.70 / 2.0, 0.31 / 2.0, 0.40 / 2.0], dtype=np.float32)

        # Point cloud coloring
        # Modes: 'auto' (use RGB if provided, else fallback),
        #        'height' (by z), 'distance' (by Euclidean distance),
        #        'rgb' (only use provided RGB), 'camera' (project camera onto points),
        #        'none' (no colors)
        self.pc_color_mode = str(p.get("pc_color_mode", "height")).lower()

        # Camera-lidar fusion for 'camera' color mode and secondary camera-colored view
        self._fusion: Optional[CameraLidarFusion] = None
        if FUSION_AVAILABLE:
            self._fusion = create_fusion(640, 480, 120.0)

        # Cache decoded RGB image for camera coloring
        self._last_img_rgb: Optional[np.ndarray] = None

        # Occupancy is handled by a separate node; this viewer only logs its output.

        # Rerun setup
        rr.init(self.rerun_app_id, spawn=self.rerun_spawn)
        rr.log("world", rr.ViewCoordinates.RDF)
        # Subscribe to topics from go2_tide_bridge
        self.subscribe("sensor/camera/front/image", self._on_image)
        # Allow overriding points topic to use cached/full map if available
        self.points_topic = str(p.get("points_topic", "sensor/lidar/points3d"))
        self.subscribe(self.points_topic, self._on_points)
        self.subscribe("state/pose3d", self._on_pose)
        # Optional correction from LidarCacheNode
        self._pose_corr = {"dx": 0.0, "dy": 0.0, "dyaw": 0.0}
        self.subscribe("mapping/pose_correction", self._on_pose_correction)
        # Optional occupancy image from separate node
        self.subscribe("mapping/occupancy/image", self._on_occ_image)
        # Standardized occupancy grid (Tide OccupancyGrid2D) and optional pre-rendered image
        self._last_occ_grid: Optional[Dict[str, Any]] = None
        self._last_occ_img: Optional[np.ndarray] = None  # RGB image if provided by producer
        self.subscribe("mapping/occupancy", self._on_occ_grid)
        # Planning overlays
        self._last_target: Optional[Dict[str, float]] = None
        self._last_path: Optional[Dict[str, Any]] = None
        self.subscribe("planning/target_pose2d", self._on_target)
        self.subscribe("planning/path", self._on_path)

        # Occupancy -> 3D obstacle overlay on the voxel/point view
        # Draw vertical boxes where 2D grid marks occupied, to indicate obstacles in 3D.
        self.occ_overlay_enabled: bool = bool(p.get("occ_overlay_enabled", True))
        self.occ_overlay_z_min: float = float(p.get("occ_overlay_z_min", 0.0))
        self.occ_overlay_z_max: float = float(p.get("occ_overlay_z_max", 1.0))
        self.occ_overlay_threshold: int = int(p.get("occ_overlay_threshold", 100))
        self.occ_overlay_max_boxes: int = int(p.get("occ_overlay_max_boxes", 3000))

    def _on_image(self, msg: Dict[str, Any]):
        self._last_img = msg

    def _on_points(self, msg: Dict[str, Any]):
        self._last_pc = msg

    def _on_pose(self, msg: Dict[str, Any]):
        # Apply optional correction on read for rendering alignment
        try:
            dx = float(self._pose_corr.get("dx", 0.0))
            dy = float(self._pose_corr.get("dy", 0.0))
            dyaw = float(self._pose_corr.get("dyaw", 0.0))
            # Clone dict to avoid mutating shared
            pose = dict(msg)
            pos = dict(pose.get("position") or {})
            ori = dict(pose.get("orientation") or {})
            px = float(pos.get("x", 0.0)) + dx
            py = float(pos.get("y", 0.0)) + dy
            pz = float(pos.get("z", 0.0))
            # Adjust yaw only; keep roll/pitch
            qw = float(ori.get("w", 1.0))
            qx = float(ori.get("x", 0.0))
            qy = float(ori.get("y", 0.0))
            qz = float(ori.get("z", 0.0))
            yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
            yaw += dyaw
            cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
            # Reconstruct quaternion with new yaw and zero roll/pitch deltas
            # Keep roll/pitch from original approximately by rotating only around Z
            # For simplicity, create yaw-only quaternion and multiply
            q_yaw = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float32)  # x,y,z,w
            # Fallback: just set orientation to yaw-only
            pose["position"] = {"x": px, "y": py, "z": pz}
            pose["orientation"] = {"w": float(q_yaw[3]), "x": float(q_yaw[0]), "y": float(q_yaw[1]), "z": float(q_yaw[2])}
            self._last_pose = pose
            self._pose_cache = pose
        except Exception:
            self._last_pose = msg
            self._pose_cache = msg

    def _on_pose_correction(self, msg: Any):
        try:
            if isinstance(msg, (bytes, bytearray)):
                import json as _json

                self._pose_corr = _json.loads(msg)
            elif isinstance(msg, str):
                self._pose_corr = json.loads(msg)
            elif isinstance(msg, dict):
                self._pose_corr = msg
        except Exception:
            pass

    def _on_occ_image(self, msg: Any):
        # Cache a pre-rendered occupancy image; log later in step to avoid competing draws
        try:
            if isinstance(msg, (bytes, bytearray)):
                payload = json.loads(msg)
            elif isinstance(msg, str):
                payload = json.loads(msg)
            elif isinstance(msg, dict):
                payload = msg
            else:
                return

            h = int(payload.get("height", 0))
            w = int(payload.get("width", 0))
            enc = str(payload.get("encoding", "rgb8")).lower()
            fmt = str(payload.get("format", "")).lower()
            data = payload.get("data")
            raw = b""
            if fmt == "json_b64":
                b64 = payload.get("data_b64", "")
                if isinstance(b64, str) and b64:
                    raw = base64.b64decode(b64)
            elif isinstance(data, (bytes, bytearray)):
                raw = data
            elif isinstance(data, str):
                raw = base64.b64decode(data)
            if h <= 0 or w <= 0 or len(raw) < h * w * 3:
                return
            img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            if enc == "bgr8":
                img = img[:, :, ::-1]
            self._last_occ_img = img
        except Exception:
            pass

    def _on_occ_grid(self, msg: Dict[str, Any]):
        # Accept dict directly, or attempt to decode bytes (CBOR/JSON) into dict
        try:
            if isinstance(msg, dict) and "width" in msg and "height" in msg and "data" in msg:
                self._last_occ_grid = msg
                return
            if isinstance(msg, (bytes, bytearray)):
                dec = None
                # Try CBOR first
                try:
                    import cbor2  # type: ignore

                    dec = cbor2.loads(msg)
                except Exception:
                    dec = None
                # Fallback to JSON
                if dec is None:
                    try:
                        import json as _json

                        dec = _json.loads(msg)
                    except Exception:
                        dec = None
                if isinstance(dec, dict) and "width" in dec and "height" in dec and "data" in dec:
                    self._last_occ_grid = dec
        except Exception:
            pass

    def _on_target(self, msg: Dict[str, Any]):
        try:
            if isinstance(msg, dict) and "x" in msg and "y" in msg:
                self._last_target = {"x": float(msg.get("x", 0.0)), "y": float(msg.get("y", 0.0))}
        except Exception:
            pass

    def _on_path(self, msg: Dict[str, Any]):
        try:
            self._last_path = msg if isinstance(msg, dict) else None
        except Exception:
            self._last_path = None

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
            # Cache RGB for camera coloring mode
            self._last_img_rgb = rgb.copy()
        except Exception:
            pass

    def _render_pointcloud(self, pc_msg: Dict[str, Any]):
        try:
            count = int(pc_msg.get("count", 0))
            xyz = pc_msg.get("xyz")
            # Support JSON fallback with base64 fields
            if not isinstance(xyz, (bytes, bytearray)) and pc_msg.get("format") == "json_b64":
                import base64

                b64 = pc_msg.get("xyz_b64")
                if isinstance(b64, str):
                    xyz = base64.b64decode(b64)
            if count <= 0 or not isinstance(xyz, (bytes, bytearray)):
                return
            pts = np.frombuffer(xyz, dtype=np.float32).reshape((-1, 3))

            colors = None

            # Try to use provided RGB if available and mode allows
            rgb_bytes = pc_msg.get("rgb")
            if not isinstance(rgb_bytes, (bytes, bytearray)) and pc_msg.get("format") == "json_b64":
                import base64

                b64r = pc_msg.get("rgb_b64")
                if isinstance(b64r, str):
                    rgb_bytes = base64.b64decode(b64r)
            if (
                self.pc_color_mode in ("auto", "rgb")
                and isinstance(rgb_bytes, (bytes, bytearray))
                and len(rgb_bytes) >= count * 3
            ):
                colors = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((-1, 3))

            # Camera mode: project camera image onto point cloud
            if colors is None and self.pc_color_mode == "camera" and self._fusion is not None:
                if self._last_img_rgb is not None:
                    try:
                        # Use fusion to colorize points with camera image
                        # Points are assumed to be in base frame (from lidar cache)
                        colors = self._fusion.colorize_pointcloud(pts, self._last_img_rgb)
                        # Points outside camera FOV get a default gray color
                        no_color = np.all(colors == 0, axis=1)
                        colors[no_color] = [80, 80, 80]  # Gray for non-visible points
                    except Exception as e:
                        print(f"Camera coloring error: {e}")
                        colors = None

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

            # Also log camera-colored version if fusion is available (even when not primary mode)
            if self._fusion is not None and self._last_img_rgb is not None and self.pc_color_mode != "camera":
                try:
                    cam_colors = self._fusion.colorize_pointcloud(pts, self._last_img_rgb)
                    # Only log if we got meaningful colors (some points visible in camera)
                    has_color = np.any(cam_colors > 0)
                    if has_color:
                        no_color = np.all(cam_colors == 0, axis=1)
                        cam_colors[no_color] = [60, 60, 60]  # Dark gray for non-visible
                        rr.log("/lidar/points_camera", rr.Points3D(pts, colors=cam_colors))
                except Exception:
                    pass

            # Occupancy grid generation is handled by OccupancyGridNode.
        except Exception:
            pass

    def _render_robot_box(self, pose_msg: Dict[str, Any]):
        try:
            pos = pose_msg.get("position") or {}
            px = float(pos.get("x", 0.0))
            py = float(pos.get("y", 0.0))
            pz = float(pos.get("z", 0.0))

            center = np.array([px, py, pz], dtype=np.float32)

            # Optional orientation (quaternion w,x,y,z)
            quat_xyzw = None
            try:
                ori = pose_msg.get("orientation") or {}
                qw = float(ori.get("w"))
                qx = float(ori.get("x"))
                qy = float(ori.get("y"))
                qz = float(ori.get("z"))
                # Rerun expects quaternions in xyzw order
                quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float32)
            except Exception:
                quat_xyzw = None

            # Log a simple axis-aligned 3D box at the robot pose.
            # Box dimensions: 0.70 x 0.31 x 0.40 m (half-sizes set in __init__).
            if quat_xyzw is not None:
                rr.log(
                    "/robot/footprint",
                    rr.Boxes3D(
                        centers=np.expand_dims(center, axis=0),
                        half_sizes=np.expand_dims(self._robot_half_sizes, axis=0),
                        quaternions=np.expand_dims(quat_xyzw, axis=0),
                    ),
                )
            else:
                rr.log(
                    "/robot/footprint",
                    rr.Boxes3D(
                        centers=np.expand_dims(center, axis=0),
                        half_sizes=np.expand_dims(self._robot_half_sizes, axis=0),
                    ),
                )
        except Exception:
            pass

    def step(self) -> None:
        if self._last_img is not None:
            self._render_image(self._last_img)
            self._last_img = None
        if self._last_pc is not None:
            self._render_pointcloud(self._last_pc)
            self._last_pc = None
        # Prefer pre-rendered occupancy image if available; fallback to grid-to-image conversion
        if self._last_occ_img is not None:
            try:
                rr.log("/occupancy/grid", rr.Image(self._last_occ_img))
                # If we also have the structured grid, overlay 3D obstacle boxes
                if self.occ_overlay_enabled and isinstance(self._last_occ_grid, dict):
                    try:
                        g = self._last_occ_grid
                        w = int(g.get("width", 0))
                        h = int(g.get("height", 0))
                        data = g.get("data")
                        if w > 0 and h > 0 and isinstance(data, list) and len(data) >= w * h:
                            arr = np.array(data[: w * h], dtype=np.int16).reshape((h, w))
                            res = float(g.get("resolution", 0.1))
                            ox = float(g.get("origin_x", 0.0))
                            oy = float(g.get("origin_y", 0.0))
                            occ_mask = arr >= int(self.occ_overlay_threshold)
                            ys, xs = np.where(occ_mask)
                            n = int(xs.shape[0])
                            if n > 0:
                                if n > self.occ_overlay_max_boxes and self.occ_overlay_max_boxes > 0:
                                    stride = int(np.ceil(n / float(self.occ_overlay_max_boxes)))
                                    xs = xs[::stride]
                                    ys = ys[::stride]
                                    n = xs.shape[0]
                                cx = ox + (xs.astype(np.float32) + 0.5) * res
                                cy = oy + (ys.astype(np.float32) + 0.5) * res
                                z0 = float(self.occ_overlay_z_min)
                                z1 = float(self.occ_overlay_z_max)
                                cz = (z0 + z1) * 0.5
                                hz = max(1e-3, (z1 - z0) * 0.5)
                                centers = np.stack([cx, cy, np.full_like(cx, cz)], axis=1).astype(np.float32)
                                half_sizes = np.tile(np.array([res * 0.5, res * 0.5, hz], dtype=np.float32), (n, 1))
                                colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (n, 1))
                                rr.log("/occupancy/obstacles3d", rr.Boxes3D(centers=centers, half_sizes=half_sizes, colors=colors))
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                self._last_occ_img = None
        elif self._last_occ_grid is not None:
            try:
                g = self._last_occ_grid
                w = int(g.get("width", 0))
                h = int(g.get("height", 0))
                data = g.get("data")
                if w > 0 and h > 0 and isinstance(data, list) and len(data) >= w * h:
                    arr = np.array(data[: w * h], dtype=np.int16).reshape((h, w))
                    # Map values to colors: unknown(-1)=gray, free(0)=green, occupied(>=100)=red
                    img = np.zeros((h, w, 3), dtype=np.uint8)
                    unknown = arr < 0
                    free = arr == 0
                    occ = arr >= 100
                    img[unknown] = (80, 80, 80)
                    img[free] = (0, 255, 0)
                    img[occ] = (255, 0, 0)
                    rr.log("/occupancy/grid", rr.Image(img))

                    # Also overlay 3D obstacles as vertical boxes at occupied cells
                    if self.occ_overlay_enabled:
                        try:
                            res = float(g.get("resolution", 0.1))
                            ox = float(g.get("origin_x", 0.0))
                            oy = float(g.get("origin_y", 0.0))
                            occ_mask = arr >= int(self.occ_overlay_threshold)
                            ys, xs = np.where(occ_mask)
                            n = int(xs.shape[0])
                            if n > 0:
                                # Optionally downsample to avoid overdraw
                                if n > self.occ_overlay_max_boxes and self.occ_overlay_max_boxes > 0:
                                    stride = int(np.ceil(n / float(self.occ_overlay_max_boxes)))
                                    xs = xs[::stride]
                                    ys = ys[::stride]
                                    n = xs.shape[0]
                                cx = ox + (xs.astype(np.float32) + 0.5) * res
                                cy = oy + (ys.astype(np.float32) + 0.5) * res
                                z0 = float(self.occ_overlay_z_min)
                                z1 = float(self.occ_overlay_z_max)
                                cz = (z0 + z1) * 0.5
                                hz = max(1e-3, (z1 - z0) * 0.5)
                                centers = np.stack([cx, cy, np.full_like(cx, cz)], axis=1).astype(np.float32)
                                half_sizes = np.tile(np.array([res * 0.5, res * 0.5, hz], dtype=np.float32), (n, 1))
                                colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (n, 1))
                                rr.log("/occupancy/obstacles3d", rr.Boxes3D(centers=centers, half_sizes=half_sizes, colors=colors))
                        except Exception:
                            pass
            except Exception:
                pass
            finally:
                self._last_occ_grid = None
        if self._last_pose is not None:
            self._render_robot_box(self._last_pose)
            self._last_pose = None
        # Render target and path last
        if self._last_target is not None:
            try:
                tx = float(self._last_target.get("x", 0.0))
                ty = float(self._last_target.get("y", 0.0))
                tz = 0.05
                rr.log("/planning/target", rr.Points3D(np.array([[tx, ty, tz]], dtype=np.float32), radii=0.06, colors=np.array([[255, 200, 0]], dtype=np.uint8)))
            except Exception:
                pass
            finally:
                self._last_target = None
        if self._last_path is not None:
            try:
                poses = self._last_path.get("poses") if isinstance(self._last_path, dict) else None
                if isinstance(poses, list) and len(poses) > 1:
                    pts = np.array([[float(p.get("x", 0.0)), float(p.get("y", 0.0)), 0.05] for p in poses], dtype=np.float32)
                    rr.log("/planning/path", rr.LineStrips3D([pts], colors=np.array([[0, 128, 255]], dtype=np.uint8)))
            except Exception:
                pass
            finally:
                self._last_path = None
