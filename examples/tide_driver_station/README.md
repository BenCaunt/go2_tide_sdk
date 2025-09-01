Go2 Tide Driver Station + Navigation

Overview
- Driver station with gamepad teleop and targeting mode
- Occupancy grid generator from LiDAR
- SE(2) A* planner with 16 heading bins
- Simple P path follower publishing `cmd/twist/nav` (muxed to `cmd/twist`)
- Rerun-based viewer with LiDAR, occupancy, target, and path overlays
  - Occupancy 2D image also overlays robot pose and planned path

Occupancy/Point Cloud Alignment
- Incoming LiDAR points are assumed to be in a consistent world frame by default.
- If your bridge publishes points in the robot/base frame, enable `pc_in_robot_frame: true` in the Occupancy node so points are transformed by the latest pose before integration.
- The grid now supports clearing previously occupied cells when free evidence appears. Tune with `clear_occupied` and `clear_free_min_points`.

Twist Mux
- Teleop and follower now publish to separate inputs: `cmd/twist/teleop` and `cmd/twist/nav`.
- `TwistMuxNode` selects between them and publishes the final `cmd/twist`.
- Policy: teleop wins only when its magnitude exceeds a small epsilon; otherwise nav passes through. Stale inputs are ignored.

Run
1) Configure your robot ID and network in `config/config.yaml`.
2) Ensure the Go2 bridge provides `state/pose3d`, `sensor/lidar/points3d`, and optionally `sensor/imu/quat`.
3) Launch the Tide session with the config (from project root):
   tide run -c config/config.yaml

Gamepad Controls
- Teleop: left stick drives (x/y), right stick yaw.
- Enable: optionally set `enable_button` to require a hold (e.g., LB=4).
- Targeting: press `target_mode_button` (default X=2) to enter targeting; move cursor with sticks; press `send_goal_button` (default A=0) to send goal.
- Cancel: any stick |axis| > `cancel_threshold` (default 0.25) cancels active navigation and returns to teleop.

Tuning
- Planner: `heading_bins`, `step_cells`, `turn_bins`, `robot_radius_m`, `allow_unknown`.
  - Turning behavior: `allow_turn_in_place` (default true) adds pure rotation steps.
  - Weights: `cost_turn_in_place` (per bin) vs `cost_turn_while_moving` (per bin while translating). Increase the latter to prefer rotate-then-straight.
- Follower: `k_dist`, `k_yaw`, `v_max`, `w_max`, `waypoint_*`, `goal_*`.
