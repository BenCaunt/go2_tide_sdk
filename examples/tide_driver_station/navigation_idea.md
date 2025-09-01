Navigation Plan (Go2 + Tide)

- Inputs: LiDAR (points3d), odometry (pose3d/imu), occupancy grid (existing node).
- Goal: Select a 2D pose and safely navigate to it with SE(2) planning and a simple follower.
- Planner: SE(2) lattice A* with 16 heading bins (360/16 deg) and footprint collision checks.
- Follower: 2x P controller (distance, heading) that drives like diff-drive; recompute path when occupancy invalidates waypoints.
- UI: Gamepad targeting mode to place a goal; show target + path in 2D and 3D; joystick override cancels.

Components

- nodes/navigation_node.py: SE(2) A* planner
  - Subscribes: `mapping/occupancy`, `state/pose3d`, `cmd/nav/goal`, `cmd/nav/cancel`
  - Publishes: `planning/path` (list of poses), `planning/target_pose2d` (echo), `planning/debug` (optional)
  - Behavior: On goal, plan from robot pose to goal with 16-theta A* and a circular footprint check. Replan if occupancy changes make any future waypoint occupied. Cancel clears current path.

- nodes/path_follow_node.py: simple P follower
  - Subscribes: `planning/path`, `state/pose3d`, `cmd/nav/cancel`
  - Publishes: `cmd/twist` (Twist2D)
  - Behavior: Track next waypoint; linear speed ∝ distance; angular speed ∝ heading error; advance waypoint when within tolerance; stop at goal. Cancel stops and clears path.

- nodes/driver_station_node.py: targeting + cancel
  - Adds targeting mode toggle (`target_mode_button`) and send goal (`send_goal_button`).
  - While targeting: move a target in XY with sticks; publish `planning/target_pose2d` for visualization; send as `cmd/nav/goal` on button press; twist output held at zero.
  - Any joystick movement with |axis| > `cancel_threshold` outside targeting publishes `cmd/nav/cancel` and returns to teleop.

- nodes/go2_sensors_node.py: viewer overlays
  - Subscribes to `planning/target_pose2d` and `planning/path`.
  - Renders target marker and path polyline in Rerun, alongside occupancy grid and robot pose.

SE(2) A* Details

- State: (ix, iy, ih) where (ix, iy) are occupancy grid cells and ih ∈ [0, 15] is heading bin; yaw = 2π ih / 16.
- Actions: Turn {-1, 0, +1} bins and move forward a fixed step (1–2 cells). Use simple motion primitive: rotate in place to next bin, then translate, collision-checking samples along the segment with a circular footprint radius R.
- Heuristic: Euclidean distance (cells) + small yaw difference penalty to prefer aligned goals.
- Collision: Occupied if any grid cell within radius R around sampled pose is ≥ 100 (occupied). Unknown allowed or configurable; default treats unknown as blocked near goal but allowed for planning except near robot.

Follower Details

- Choose next waypoint ahead by index; compute vector (dx, dy) to it; desired yaw = atan2(dy, dx).
- Errors: e_dist = hypot(dx, dy), e_yaw = wrap(angle_diff(desired_yaw - yaw)).
- Commands: v = clamp(k_dist * e_dist, 0, v_max), wz = clamp(k_yaw * e_yaw, -w_max, w_max).
- Advance waypoint when e_dist < d_tol; finish when last waypoint within goal_tol and |e_yaw| < yaw_tol.
- Replan trigger: navigation_node handles, follower just follows current path and stops on cancel.

Gamepad UX

- target_mode_button toggles targeting; initializes target at current robot pose.
- While targeting: left/right stick moves target in world XY at a configurable rate; optional yaw adjust with right stick.
- send_goal_button publishes `cmd/nav/goal` with current target and exits targeting.
- Cancel: if any absolute axis value > cancel_threshold (default 0.25) outside targeting, publish `cmd/nav/cancel` and allow teleop.

Safety and Maintainability

- Small, single-purpose nodes; typed messages where available; plain dicts otherwise.
- Conservative defaults: low speeds, footprint radius ~0.25–0.30 m; treat high-confidence occupied strictly.
- DRY helpers for grid/world transforms and collision checks.
- Parameters exposed via node config to tune bin count, step, radii, gains, and thresholds.

Topics (summary)

- In: `mapping/occupancy`, `state/pose3d`, `sensor/imu/quat`
- UI: `planning/target_pose2d` (target cursor), `cmd/nav/goal`, `cmd/nav/cancel`
- Out: `planning/path`, `cmd/twist`

Next Steps

- Implement navigation_node.py and path_follow_node.py.
- Update driver_station_node.py for targeting and cancel + config.
- Update go2_sensors_node.py to draw target + path.
