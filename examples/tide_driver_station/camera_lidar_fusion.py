#!/usr/bin/env python3
"""
Camera-LiDAR Fusion Module for Unitree Go2.

Provides transforms between camera pixels and world coordinates using:
- Camera intrinsics (estimated from FOV)
- Extrinsic transforms from Go2 URDF
- LiDAR point cloud for depth estimation

Coordinate frames:
- World/Odom: Robot's odometry frame (x=forward, y=left, z=up)
- Base: Robot body frame at center
- Camera: Front camera frame (z=forward into scene, x=right, y=down)
- LiDAR: L1 lidar frame

URDF Transforms (base frame):
- Camera: position=[0.32715, -0.00003, 0.04297], rpy=[0,0,0]
- LiDAR:  position=[0.28945, 0, -0.046825], rpy=[0, 2.8782, 0]
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    width: int = 640
    height: int = 480
    fx: float = 185.0  # Focal length x (pixels) - estimated from 120° FOV
    fy: float = 185.0  # Focal length y (pixels)
    cx: float = 320.0  # Principal point x
    cy: float = 240.0  # Principal point y

    @classmethod
    def from_fov(cls, width: int, height: int, fov_horizontal_deg: float) -> "CameraIntrinsics":
        """Create intrinsics from horizontal FOV."""
        fov_rad = np.radians(fov_horizontal_deg)
        fx = width / (2.0 * np.tan(fov_rad / 2.0))
        fy = fx  # Assume square pixels
        cx = width / 2.0
        cy = height / 2.0
        return cls(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    @property
    def K(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass
class Transform:
    """Rigid 3D transform (rotation + translation)."""
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector

    @classmethod
    def from_rpy_xyz(cls, roll: float, pitch: float, yaw: float,
                     x: float, y: float, z: float) -> "Transform":
        """Create transform from roll-pitch-yaw angles and translation."""
        # Rotation matrices for each axis
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Combined rotation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ], dtype=np.float64)

        t = np.array([x, y, z], dtype=np.float64)
        return cls(R=R, t=t)

    @classmethod
    def identity(cls) -> "Transform":
        """Identity transform."""
        return cls(R=np.eye(3), t=np.zeros(3))

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply transform to points. points: (N,3) or (3,)"""
        if points.ndim == 1:
            return self.R @ points + self.t
        return (self.R @ points.T).T + self.t

    def inverse(self) -> "Transform":
        """Inverse transform."""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Transform(R=R_inv, t=t_inv)

    def compose(self, other: "Transform") -> "Transform":
        """Compose transforms: self @ other."""
        R_new = self.R @ other.R
        t_new = self.R @ other.t + self.t
        return Transform(R=R_new, t=t_new)


class Go2Transforms:
    """
    Go2 robot coordinate transforms based on URDF.

    All transforms are FROM the named frame TO the base frame.
    """

    def __init__(self):
        # Camera to base transform
        # URDF: position=[0.32715, -0.00003, 0.04297], rpy=[0,0,0]
        # Camera convention: z forward, x right, y down
        # Robot convention: x forward, y left, z up
        # Need to rotate camera frame to align with robot convention

        # Camera optical frame to camera link (standard ROS convention)
        # optical: z forward, x right, y down -> link: x forward, y left, z up
        R_optical_to_link = np.array([
            [0, 0, 1],   # camera z (forward) -> robot x
            [-1, 0, 0],  # camera x (right) -> robot -y (left)
            [0, -1, 0]   # camera y (down) -> robot -z (up)
        ], dtype=np.float64)

        # Camera link position in base frame
        t_camera = np.array([0.32715, -0.00003, 0.04297])

        # Full camera (optical) to base transform
        self.T_camera_to_base = Transform(R=R_optical_to_link, t=t_camera)
        self.T_base_to_camera = self.T_camera_to_base.inverse()

        # LiDAR to base transform
        # URDF: position=[0.28945, 0, -0.046825], rpy=[0, 2.8782, 0]
        # Note: pitch of 2.8782 rad ≈ 165° (points mostly backward/down)
        self.T_lidar_to_base = Transform.from_rpy_xyz(
            roll=0, pitch=2.8782, yaw=0,
            x=0.28945, y=0, z=-0.046825
        )
        self.T_base_to_lidar = self.T_lidar_to_base.inverse()

    def camera_to_base(self, points: np.ndarray) -> np.ndarray:
        """Transform points from camera frame to base frame."""
        return self.T_camera_to_base.apply(points)

    def base_to_camera(self, points: np.ndarray) -> np.ndarray:
        """Transform points from base frame to camera frame."""
        return self.T_base_to_camera.apply(points)

    def lidar_to_base(self, points: np.ndarray) -> np.ndarray:
        """Transform points from lidar frame to base frame."""
        return self.T_lidar_to_base.apply(points)

    def base_to_world(self, points: np.ndarray, robot_x: float, robot_y: float,
                      robot_yaw: float) -> np.ndarray:
        """Transform points from base frame to world frame."""
        T_base_to_world = Transform.from_rpy_xyz(0, 0, robot_yaw, robot_x, robot_y, 0)
        return T_base_to_world.apply(points)

    def world_to_base(self, points: np.ndarray, robot_x: float, robot_y: float,
                      robot_yaw: float) -> np.ndarray:
        """Transform points from world frame to base frame."""
        T_base_to_world = Transform.from_rpy_xyz(0, 0, robot_yaw, robot_x, robot_y, 0)
        return T_base_to_world.inverse().apply(points)


class CameraLidarFusion:
    """
    Fuses camera and LiDAR data for pixel-to-world projection.
    """

    def __init__(self,
                 intrinsics: Optional[CameraIntrinsics] = None,
                 transforms: Optional[Go2Transforms] = None):
        self.intrinsics = intrinsics or CameraIntrinsics.from_fov(640, 480, 120.0)
        self.transforms = transforms or Go2Transforms()

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """
        Convert pixel coordinates to a unit ray in camera frame.

        Args:
            u: pixel x coordinate (0 = left edge)
            v: pixel y coordinate (0 = top edge)

        Returns:
            Unit ray direction in camera optical frame (z forward)
        """
        K = self.intrinsics
        x = (u - K.cx) / K.fx
        y = (v - K.cy) / K.fy
        z = 1.0
        ray = np.array([x, y, z], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def find_depth_from_lidar(self,
                               ray_camera: np.ndarray,
                               lidar_points_base: np.ndarray,
                               angular_tolerance_deg: float = 3.0,
                               min_depth: float = 0.3,
                               max_depth: float = 10.0) -> Optional[float]:
        """
        Find depth along a camera ray using LiDAR points.

        Args:
            ray_camera: Unit ray in camera optical frame
            lidar_points_base: LiDAR points in base frame (N, 3)
            angular_tolerance_deg: Max angle between ray and lidar point
            min_depth: Minimum valid depth (m)
            max_depth: Maximum valid depth (m)

        Returns:
            Depth in meters along the ray, or None if no match found
        """
        if lidar_points_base is None or len(lidar_points_base) == 0:
            return None

        # Transform lidar points to camera frame
        lidar_points_cam = self.transforms.base_to_camera(lidar_points_base)

        # Filter points in front of camera (z > 0)
        in_front = lidar_points_cam[:, 2] > min_depth
        if not np.any(in_front):
            return None

        pts_front = lidar_points_cam[in_front]

        # Compute depth (distance along z for each point)
        depths = pts_front[:, 2]

        # Filter by depth range
        valid_depth = (depths >= min_depth) & (depths <= max_depth)
        if not np.any(valid_depth):
            return None

        pts_valid = pts_front[valid_depth]
        depths_valid = depths[valid_depth]

        # Normalize points to rays
        norms = np.linalg.norm(pts_valid, axis=1, keepdims=True)
        pts_rays = pts_valid / norms

        # Compute angular distance to target ray
        dots = pts_rays @ ray_camera
        angles_rad = np.arccos(np.clip(dots, -1.0, 1.0))
        angles_deg = np.degrees(angles_rad)

        # Find points within angular tolerance
        within_tol = angles_deg < angular_tolerance_deg
        if not np.any(within_tol):
            return None

        # Return depth of closest point (smallest angle)
        best_idx = np.argmin(angles_deg[within_tol])
        candidate_depths = depths_valid[within_tol]

        return float(candidate_depths[best_idx])

    def pixel_to_base(self,
                      u: float, v: float,
                      depth: float) -> np.ndarray:
        """
        Project pixel to 3D point in base frame given depth.

        Args:
            u, v: Pixel coordinates
            depth: Depth in meters

        Returns:
            3D point in base frame
        """
        K = self.intrinsics

        # Pixel to camera 3D (optical frame: z forward)
        x_cam = depth * (u - K.cx) / K.fx
        y_cam = depth * (v - K.cy) / K.fy
        z_cam = depth
        point_camera = np.array([x_cam, y_cam, z_cam])

        # Camera to base
        point_base = self.transforms.camera_to_base(point_camera)
        return point_base

    def pixel_to_world(self,
                       u: float, v: float,
                       lidar_points_base: np.ndarray,
                       robot_x: float, robot_y: float, robot_yaw: float,
                       fallback_depth: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Project pixel to world coordinates using LiDAR for depth.

        Args:
            u, v: Pixel coordinates
            lidar_points_base: LiDAR points in base frame
            robot_x, robot_y, robot_yaw: Robot pose in world frame
            fallback_depth: Depth to use if LiDAR match not found

        Returns:
            Tuple of (world_point, depth) or (None, None) if projection failed
        """
        # Get ray in camera frame
        ray = self.pixel_to_ray(u, v)

        # Find depth from LiDAR
        depth = self.find_depth_from_lidar(ray, lidar_points_base)

        if depth is None:
            if fallback_depth is not None:
                depth = fallback_depth
            else:
                return None, None

        # Project to base frame
        point_base = self.pixel_to_base(u, v, depth)

        # Transform to world
        point_world = self.transforms.base_to_world(point_base, robot_x, robot_y, robot_yaw)

        return point_world, depth

    def world_to_pixel(self,
                       point_world: np.ndarray,
                       robot_x: float, robot_y: float, robot_yaw: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Project world point to pixel coordinates.

        Args:
            point_world: 3D point in world frame
            robot_x, robot_y, robot_yaw: Robot pose

        Returns:
            Tuple of (u, v, depth) or (None, None, None) if behind camera
        """
        # World to base
        point_base = self.transforms.world_to_base(point_world, robot_x, robot_y, robot_yaw)

        # Base to camera
        point_cam = self.transforms.base_to_camera(point_base)

        # Check if in front of camera
        if point_cam[2] <= 0:
            return None, None, None

        # Project to pixel
        K = self.intrinsics
        u = K.fx * point_cam[0] / point_cam[2] + K.cx
        v = K.fy * point_cam[1] / point_cam[2] + K.cy

        # Check if in image bounds
        if not (0 <= u < K.width and 0 <= v < K.height):
            return None, None, None

        return float(u), float(v), float(point_cam[2])

    def colorize_pointcloud(self,
                            lidar_points_base: np.ndarray,
                            image_rgb: np.ndarray,
                            robot_x: float = 0, robot_y: float = 0, robot_yaw: float = 0) -> np.ndarray:
        """
        Color LiDAR points using camera image.

        Args:
            lidar_points_base: (N, 3) points in base frame
            image_rgb: (H, W, 3) RGB image
            robot_x, robot_y, robot_yaw: Robot pose (for points already in world frame)

        Returns:
            (N, 3) RGB colors for each point (uint8)
        """
        N = len(lidar_points_base)
        colors = np.zeros((N, 3), dtype=np.uint8)

        # Transform points to camera frame
        points_cam = self.transforms.base_to_camera(lidar_points_base)

        # Filter points in front of camera
        in_front = points_cam[:, 2] > 0.1

        if not np.any(in_front):
            return colors

        # Project to image plane
        K = self.intrinsics
        pts = points_cam[in_front]

        u = (K.fx * pts[:, 0] / pts[:, 2] + K.cx).astype(np.int32)
        v = (K.fy * pts[:, 1] / pts[:, 2] + K.cy).astype(np.int32)

        # Check bounds
        h, w = image_rgb.shape[:2]
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        # Sample colors
        valid_indices = np.where(in_front)[0][valid]
        colors[valid_indices] = image_rgb[v[valid], u[valid]]

        return colors


# Convenience function for the common use case
def create_fusion(image_width: int = 640,
                  image_height: int = 480,
                  fov_deg: float = 120.0) -> CameraLidarFusion:
    """Create a CameraLidarFusion instance with Go2 parameters."""
    intrinsics = CameraIntrinsics.from_fov(image_width, image_height, fov_deg)
    transforms = Go2Transforms()
    return CameraLidarFusion(intrinsics, transforms)
