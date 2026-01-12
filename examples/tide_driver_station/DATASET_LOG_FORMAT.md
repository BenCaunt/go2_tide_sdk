# Dataset Log Output Specification

This document describes the file layout and schema produced by
`DatasetLoggerNode`.

## Directory Layout

Each recording writes a new directory under `logs/datasets/` by default:

```
logs/datasets/<robot_id>_<timestamp>/
  samples.jsonl
  session.json
  images/
    <timestamp_ns>.<ext>
```

The output root can be overridden with `output_root` or `output_dir` in
`config/config.yaml`.

## `session.json`

Metadata captured once at startup.

Fields:

- `robot_id`: Robot identifier string.
- `start_time_ns`: Logger start time in nanoseconds.
- `image_topic`: Image topic name.
- `pose_topic`: Pose topic name.
- `twist_topic`: Twist topic name.
- `image_format`: Image file format string (`npy`, `jpg`, or `png`).

Example:

```json
{
  "robot_id": "go2",
  "start_time_ns": 1733957647123456789,
  "image_topic": "sensor/camera/front/image",
  "pose_topic": "state/pose3d",
  "twist_topic": "cmd/twist",
  "image_format": "npy"
}
```

## `samples.jsonl`

Newline-delimited JSON. Each line corresponds to a single synchronized sample
and is appended in capture order.

Fields:

- `timestamp_ns`: Image capture timestamp in nanoseconds.
- `robot_id`: Robot identifier string.
- `pose`: Pose dictionary with:
  - `position`: `{x, y, z}` in meters.
  - `orientation`: `{w, x, y, z}` quaternion.
  - `frame_id` (optional): Coordinate frame string.
- `cmd_twist`: Commanded velocity with `vx`, `vy` (m/s), `wz` (rad/s).
- `image`: Image metadata with:
  - `path`: Relative path to the image file (e.g., `images/<timestamp_ns>.<ext>`).
  - `height`, `width`: Image dimensions in pixels.
  - `format`: File format string (`npy`, `jpg`, `png`).
  - `stored_encoding`: Encoding of the stored pixels (`bgr8` or `rgb8`).
  - `source_encoding`: Original encoding from the incoming camera message.

Example:

```json
{
  "timestamp_ns": 1733957648123456789,
  "robot_id": "go2",
  "pose": {
    "position": {"x": 1.2, "y": -0.4, "z": 0.0},
    "orientation": {"w": 0.98, "x": 0.0, "y": 0.0, "z": 0.2},
    "frame_id": "map"
  },
  "cmd_twist": {"vx": 0.1, "vy": 0.0, "wz": 0.05},
  "image": {
    "path": "images/1733957648123456789.npy",
    "height": 720,
    "width": 1280,
    "format": "npy",
    "stored_encoding": "bgr8",
    "source_encoding": "bgr8"
  }
}
```

## Image Files

- Filenames are derived from the sample `timestamp_ns`.
- `npy` images store raw arrays (no compression).
- `jpg` and `png` are encoded using OpenCV (if available).
