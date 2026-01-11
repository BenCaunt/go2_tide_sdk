#!/usr/bin/env python3
"""
Go2 MCP Server - Model Context Protocol server for Unitree Go2 robot control.

This server exposes tools for:
- Getting camera images
- Getting robot pose
- Moving the robot (velocity commands)
- Navigation to waypoints

Run with: python mcp_server.py
Or install in Claude Desktop config.
"""

import asyncio
import base64
import json
import time
import threading
from typing import Any, Optional
from dataclasses import dataclass, field

try:
    from fastmcp import FastMCP
except ImportError:
    print("FastMCP not installed. Install with: pip install fastmcp")
    raise

try:
    import zenoh
except ImportError:
    print("Zenoh not installed. Install with: pip install eclipse-zenoh")
    raise

import numpy as np

from camera_lidar_fusion import CameraLidarFusion, create_fusion


@dataclass
class RobotState:
    """Cached robot state from Zenoh subscriptions."""
    last_image: Optional[dict] = None
    last_image_time: float = 0.0
    last_pose: Optional[dict] = None
    last_pose_time: float = 0.0
    last_lidar: Optional[dict] = None
    last_lidar_time: float = 0.0
    last_occupancy: Optional[dict] = None
    last_occupancy_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)


class Go2ZenohClient:
    """Zenoh client for communicating with Go2 robot via Tide topics."""

    def __init__(self, robot_id: str = "cash"):
        self.robot_id = robot_id
        self.session: Optional[zenoh.Session] = None
        self.state = RobotState()
        self._subscribers: list = []
        self._running = False

    def connect(self):
        """Connect to Zenoh and subscribe to robot topics."""
        config = zenoh.Config()
        self.session = zenoh.open(config)
        self._running = True

        # Subscribe to robot topics with callbacks
        topics = [
            (f"{self.robot_id}/sensor/camera/front/image", self._on_image),
            (f"{self.robot_id}/state/pose3d", self._on_pose),
            (f"{self.robot_id}/sensor/lidar/points3d_cached", self._on_lidar),
            (f"{self.robot_id}/mapping/occupancy", self._on_occupancy),
        ]

        for topic, callback in topics:
            sub = self.session.declare_subscriber(topic, callback)
            self._subscribers.append(sub)

        print(f"Go2ZenohClient: Connected and subscribed to {len(topics)} topics")

    def _on_image(self, sample: zenoh.Sample):
        """Handle incoming camera image."""
        try:
            payload = sample.payload.to_bytes()
            # Try to decode as JSON first
            try:
                data = json.loads(payload)
            except:
                # Binary format - store raw
                data = {"raw": payload, "format": "binary"}
            with self.state._lock:
                self.state.last_image = data
                self.state.last_image_time = time.time()
        except Exception as e:
            print(f"Error processing image: {e}")

    def _on_pose(self, sample: zenoh.Sample):
        """Handle incoming pose."""
        try:
            payload = sample.payload.to_bytes()
            try:
                data = json.loads(payload)
            except:
                # Try CBOR
                try:
                    import cbor2
                    data = cbor2.loads(payload)
                except:
                    data = None
            if data:
                with self.state._lock:
                    self.state.last_pose = data
                    self.state.last_pose_time = time.time()
        except Exception as e:
            print(f"Error processing pose: {e}")

    def _on_lidar(self, sample: zenoh.Sample):
        """Handle incoming lidar points."""
        try:
            payload = sample.payload.to_bytes()
            try:
                data = json.loads(payload)
            except:
                try:
                    import cbor2
                    data = cbor2.loads(payload)
                except:
                    data = {"raw": payload}
            with self.state._lock:
                self.state.last_lidar = data
                self.state.last_lidar_time = time.time()
        except Exception as e:
            print(f"Error processing lidar: {e}")

    def _on_occupancy(self, sample: zenoh.Sample):
        """Handle incoming occupancy grid."""
        try:
            payload = sample.payload.to_bytes()
            try:
                data = json.loads(payload)
            except:
                try:
                    import cbor2
                    data = cbor2.loads(payload)
                except:
                    data = None
            if data:
                with self.state._lock:
                    self.state.last_occupancy = data
                    self.state.last_occupancy_time = time.time()
        except Exception as e:
            print(f"Error processing occupancy: {e}")

    def publish(self, topic: str, data: Any):
        """Publish data to a Zenoh topic."""
        if self.session is None:
            raise RuntimeError("Not connected to Zenoh")
        full_topic = f"{self.robot_id}/{topic}"
        payload = json.dumps(data).encode()
        self.session.put(full_topic, payload)

    def close(self):
        """Clean up Zenoh resources."""
        self._running = False
        for sub in self._subscribers:
            sub.undeclare()
        if self.session:
            self.session.close()


# Global client instance
_client: Optional[Go2ZenohClient] = None


def get_client() -> Go2ZenohClient:
    """Get or create the global Zenoh client."""
    global _client
    if _client is None:
        _client = Go2ZenohClient(robot_id="cash")
    return _client


# Initialize camera-lidar fusion
_fusion: Optional[CameraLidarFusion] = None


def get_fusion() -> CameraLidarFusion:
    """Get or create the global fusion instance."""
    global _fusion
    if _fusion is None:
        _fusion = create_fusion(640, 480, 120.0)
    return _fusion


def decode_lidar_points(lidar_data: Optional[dict]) -> Optional[np.ndarray]:
    """Decode lidar points from cached message to numpy array (N, 3)."""
    if lidar_data is None:
        return None

    try:
        # Check for xyz field (binary float32)
        xyz = lidar_data.get("xyz")
        if xyz is None and lidar_data.get("format") == "json_b64":
            xyz_b64 = lidar_data.get("xyz_b64")
            if xyz_b64:
                xyz = base64.b64decode(xyz_b64)

        if xyz is None:
            # Raw format
            raw = lidar_data.get("raw")
            if raw and isinstance(raw, (bytes, bytearray)):
                # Assume raw is xyz float32 data
                xyz = raw

        if xyz is not None and isinstance(xyz, (bytes, bytearray)):
            points = np.frombuffer(xyz, dtype=np.float32).reshape(-1, 3)
            return points

        return None
    except Exception as e:
        print(f"Error decoding lidar: {e}")
        return None


# Initialize FastMCP server
mcp = FastMCP("go2-mcp-server")


@mcp.tool()
async def get_camera_image() -> dict:
    """
    Get the current camera image from the Go2 robot's front camera.

    Returns a dict with:
    - width: image width in pixels
    - height: image height in pixels
    - encoding: pixel format (e.g., 'bgr8', 'rgb8')
    - data_b64: base64 encoded image data
    - timestamp: when the image was captured
    - age_seconds: how old the image is

    The image can be decoded and analyzed to identify objects,
    read text, or understand the robot's environment.
    """
    client = get_client()

    if client.state.last_image is None:
        return {"error": "No camera image available yet. Is the robot connected?"}

    img = client.state.last_image
    age = time.time() - client.state.last_image_time

    # Handle different image formats
    if "raw" in img:
        # Binary format
        return {
            "data_b64": base64.b64encode(img["raw"]).decode(),
            "format": "binary",
            "age_seconds": round(age, 2),
        }

    # JSON format with image data
    result = {
        "width": img.get("width", 0),
        "height": img.get("height", 0),
        "encoding": img.get("encoding", "unknown"),
        "age_seconds": round(age, 2),
    }

    # Include base64 image data
    if "data" in img and isinstance(img["data"], (bytes, bytearray)):
        result["data_b64"] = base64.b64encode(img["data"]).decode()
    elif "data_b64" in img:
        result["data_b64"] = img["data_b64"]
    elif "data" in img and isinstance(img["data"], str):
        result["data_b64"] = img["data"]

    return result


@mcp.tool()
async def get_robot_pose() -> dict:
    """
    Get the current pose (position and orientation) of the Go2 robot.

    Returns a dict with:
    - x: position in meters (forward from start)
    - y: position in meters (left from start)
    - z: height in meters
    - yaw: heading angle in radians (-pi to pi, 0 = forward)
    - yaw_degrees: heading angle in degrees
    - timestamp: when pose was captured
    - age_seconds: how old the pose data is

    The pose is in the robot's odometry frame, relative to where it started.
    """
    client = get_client()

    if client.state.last_pose is None:
        return {"error": "No pose data available yet. Is the robot connected?"}

    pose = client.state.last_pose
    age = time.time() - client.state.last_pose_time

    # Extract position
    pos = pose.get("position", {})
    x = float(pos.get("x", 0.0))
    y = float(pos.get("y", 0.0))
    z = float(pos.get("z", 0.0))

    # Extract orientation (quaternion) and convert to yaw
    ori = pose.get("orientation", {})
    qw = float(ori.get("w", 1.0))
    qx = float(ori.get("x", 0.0))
    qy = float(ori.get("y", 0.0))
    qz = float(ori.get("z", 0.0))

    # Convert quaternion to yaw (rotation about Z axis)
    yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "z": round(z, 3),
        "yaw": round(yaw, 4),
        "yaw_degrees": round(np.degrees(yaw), 1),
        "age_seconds": round(age, 2),
    }


@mcp.tool()
async def move(
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    duration: float = 0.5
) -> dict:
    """
    Move the robot with velocity commands.

    Args:
        vx: Forward velocity in m/s (positive = forward, negative = backward). Max ~1.0 m/s.
        vy: Lateral velocity in m/s (positive = left, negative = right). Max ~0.5 m/s.
        wz: Angular velocity in rad/s (positive = turn left, negative = turn right). Max ~2.0 rad/s.
        duration: How long to apply this velocity in seconds (default 0.5s).

    Returns status of the command.

    Safety notes:
    - Commands are automatically limited to safe velocities
    - The robot will stop after the duration expires
    - Use stop() for immediate halt
    """
    client = get_client()

    # Safety limits
    MAX_VX = 1.0
    MAX_VY = 0.5
    MAX_WZ = 2.0
    MAX_DURATION = 5.0

    # Clamp values
    vx = max(-MAX_VX, min(MAX_VX, vx))
    vy = max(-MAX_VY, min(MAX_VY, vy))
    wz = max(-MAX_WZ, min(MAX_WZ, wz))
    duration = max(0.1, min(MAX_DURATION, duration))

    # Publish velocity command
    twist_msg = {
        "linear": {"x": vx, "y": vy},
        "angular": wz
    }

    try:
        # Send command for duration (run in thread to not block)
        def send_commands():
            start = time.time()
            while time.time() - start < duration:
                client.publish("cmd/twist/teleop", twist_msg)
                time.sleep(0.05)  # 20Hz
            # Stop after duration
            client.publish("cmd/twist/teleop", {
                "linear": {"x": 0.0, "y": 0.0},
                "angular": 0.0
            })

        await asyncio.get_event_loop().run_in_executor(None, send_commands)

        return {
            "status": "completed",
            "vx": vx,
            "vy": vy,
            "wz": wz,
            "duration": duration,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def stop() -> dict:
    """
    Immediately stop all robot motion.

    Sends zero velocity commands and cancels any active navigation.
    Use this for emergency stops or to halt the robot quickly.
    """
    client = get_client()

    try:
        # Send stop command
        stop_msg = {
            "linear": {"x": 0.0, "y": 0.0},
            "angular": 0.0
        }
        client.publish("cmd/twist/teleop", stop_msg)

        # Also cancel any active navigation
        client.publish("cmd/nav/cancel", {"reason": "mcp_stop"})

        return {"status": "stopped"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def navigate_to(x: float, y: float, yaw: float = 0.0) -> dict:
    """
    Navigate the robot to a target position using the onboard path planner.

    Args:
        x: Target X position in meters (in odometry frame)
        y: Target Y position in meters (in odometry frame)
        yaw: Target heading in radians (optional, default 0)

    The robot will plan a collision-free path using its occupancy grid
    and follow it autonomously. Use cancel_navigation() to abort.

    Returns the navigation goal that was sent.
    """
    client = get_client()

    goal_msg = {
        "x": float(x),
        "y": float(y),
        "yaw": float(yaw),
    }

    try:
        client.publish("cmd/nav/goal", goal_msg)
        return {
            "status": "goal_sent",
            "target": goal_msg,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def cancel_navigation() -> dict:
    """
    Cancel any active navigation goal.

    The robot will stop following its current path and halt.
    """
    client = get_client()

    try:
        client.publish("cmd/nav/cancel", {"reason": "mcp_cancel"})
        return {"status": "navigation_cancelled"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def get_occupancy_grid() -> dict:
    """
    Get the current 2D occupancy grid map built from lidar.

    Returns:
    - width, height: grid dimensions in cells
    - resolution: meters per cell
    - origin_x, origin_y: world coordinates of grid origin
    - data: flattened grid values (-1=unknown, 0=free, >=100=occupied)
    - age_seconds: how old the map is

    Use this to understand what obstacles the robot sees around it.
    """
    client = get_client()

    if client.state.last_occupancy is None:
        return {"error": "No occupancy grid available yet."}

    occ = client.state.last_occupancy
    age = time.time() - client.state.last_occupancy_time

    return {
        "width": occ.get("width", 0),
        "height": occ.get("height", 0),
        "resolution": occ.get("resolution", 0.1),
        "origin_x": occ.get("origin_x", 0.0),
        "origin_y": occ.get("origin_y", 0.0),
        "data_length": len(occ.get("data", [])),
        "age_seconds": round(age, 2),
        # Don't include full data array - too large. Summarize instead.
        "summary": _summarize_occupancy(occ),
    }


def _summarize_occupancy(occ: dict) -> dict:
    """Generate a summary of the occupancy grid."""
    data = occ.get("data", [])
    if not data:
        return {"error": "empty grid"}

    arr = np.array(data)
    unknown = int(np.sum(arr < 0))
    free = int(np.sum(arr == 0))
    occupied = int(np.sum(arr >= 100))
    total = len(arr)

    return {
        "total_cells": total,
        "unknown_cells": unknown,
        "free_cells": free,
        "occupied_cells": occupied,
        "unknown_pct": round(100 * unknown / total, 1) if total > 0 else 0,
        "free_pct": round(100 * free / total, 1) if total > 0 else 0,
        "occupied_pct": round(100 * occupied / total, 1) if total > 0 else 0,
    }


@mcp.tool()
async def pixel_to_world(u: float, v: float, fallback_depth: float = 2.0) -> dict:
    """
    Convert a pixel coordinate to world coordinates using camera-lidar fusion.

    This is the key tool for object navigation: when you see an object in the
    camera image, use this tool to find where it is in the world so the robot
    can navigate to it.

    Args:
        u: Pixel X coordinate (0 = left edge, increases right). Image is 640 wide.
        v: Pixel Y coordinate (0 = top edge, increases down). Image is 480 tall.
        fallback_depth: Depth in meters to use if lidar doesn't have coverage (default 2.0m)

    Returns a dict with:
    - x, y, z: World coordinates in meters (robot's odometry frame)
    - depth: Estimated depth from lidar (or fallback value)
    - depth_source: 'lidar' if depth came from lidar, 'fallback' otherwise
    - robot_pose: Current robot position when projection was made

    Example workflow:
    1. Call get_camera_image() and analyze it to find an object
    2. Identify the pixel coordinates (u, v) of the object's base/center
    3. Call pixel_to_world(u, v) to get world coordinates
    4. Call navigate_to(x, y) to send the robot there
    """
    client = get_client()
    fusion = get_fusion()

    # Get current robot pose
    if client.state.last_pose is None:
        return {"error": "No robot pose available. Is the robot connected?"}

    pose = client.state.last_pose
    pos = pose.get("position", {})
    ori = pose.get("orientation", {})

    robot_x = float(pos.get("x", 0.0))
    robot_y = float(pos.get("y", 0.0))

    # Extract yaw from quaternion
    qw = float(ori.get("w", 1.0))
    qx = float(ori.get("x", 0.0))
    qy = float(ori.get("y", 0.0))
    qz = float(ori.get("z", 0.0))
    robot_yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    # Get lidar points
    lidar_points = decode_lidar_points(client.state.last_lidar)

    # Try projection with lidar depth
    depth_source = "lidar"
    point_world, depth = fusion.pixel_to_world(
        u, v,
        lidar_points if lidar_points is not None else np.array([]).reshape(0, 3),
        robot_x, robot_y, robot_yaw,
        fallback_depth=None  # Don't use fallback yet, check if lidar worked
    )

    # If lidar didn't work, use fallback
    if point_world is None:
        depth_source = "fallback"
        point_world, depth = fusion.pixel_to_world(
            u, v,
            np.array([]).reshape(0, 3),  # Empty, will use fallback
            robot_x, robot_y, robot_yaw,
            fallback_depth=fallback_depth
        )

    if point_world is None:
        return {"error": "Could not project pixel to world coordinates"}

    return {
        "x": round(float(point_world[0]), 3),
        "y": round(float(point_world[1]), 3),
        "z": round(float(point_world[2]), 3),
        "depth": round(float(depth), 3),
        "depth_source": depth_source,
        "pixel": {"u": u, "v": v},
        "robot_pose": {
            "x": round(robot_x, 3),
            "y": round(robot_y, 3),
            "yaw": round(robot_yaw, 4),
        },
    }


@mcp.tool()
async def get_lidar_points(max_points: int = 1000) -> dict:
    """
    Get a sample of the current lidar point cloud.

    Args:
        max_points: Maximum number of points to return (default 1000)

    Returns:
    - count: Number of points returned
    - points: List of [x, y, z] coordinates in robot base frame
    - age_seconds: How old the data is

    Points are in the robot's base frame (x=forward, y=left, z=up).
    """
    client = get_client()

    if client.state.last_lidar is None:
        return {"error": "No lidar data available yet."}

    points = decode_lidar_points(client.state.last_lidar)
    age = time.time() - client.state.last_lidar_time

    if points is None or len(points) == 0:
        return {"error": "Could not decode lidar data"}

    # Subsample if needed
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    return {
        "count": len(points),
        "points": [[round(p[0], 3), round(p[1], 3), round(p[2], 3)] for p in points],
        "age_seconds": round(age, 2),
    }


# Lifecycle hooks
@mcp.on_startup()
async def startup():
    """Initialize Zenoh connection on server startup."""
    client = get_client()
    # Run sync connect in executor
    await asyncio.get_event_loop().run_in_executor(None, client.connect)
    print("Go2 MCP Server started and connected to robot")


@mcp.on_shutdown()
async def shutdown():
    """Clean up on server shutdown."""
    global _client
    if _client:
        await asyncio.get_event_loop().run_in_executor(None, _client.close)
        _client = None
    print("Go2 MCP Server shut down")


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
