#!/usr/bin/env python3
"""
Async Tide <-> Unitree Go2 bridge.

Capabilities:
- Subscribes to Tide `cmd/twist` (Twist2D) and sends Go2 SPORT Move.
- Subscribes to Go2 LF_SPORT_MOD_STATE and publishes Tide Pose3D + IMU Quaternion.

Notes:
- Runs as a standalone async program (not a Tide BaseNode) per requirements.
- Uses Tide models + serialization helpers to ensure correct types over Zenoh.
"""

import os
import sys
import asyncio
import logging
import time
import threading
from typing import Optional, Dict, Any, Tuple

# Tide imports (models, topics, serialization)
from tide.namespaces import robot_topic, CmdTopic, StateTopic
from tide.models import Pose3D, Quaternion, Vector3, Twist2D
from tide.models.serialization import to_zenoh_value, from_zenoh_value

# Zenoh
try:
    import zenoh
except Exception as e:
    print("Error: zenoh package not found. Install with: pip install eclipse-zenoh")
    raise

# Go2 WebRTC driver
try:
    from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
    from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
    # Optional discovery helper
    try:
        from go2_webrtc_driver.multicast_scanner import discover_ip_sn as _discover_ip_sn
    except Exception:
        _discover_ip_sn = None
except Exception as e:
    print("Error: go2_webrtc_driver not available in workspace")
    raise


log = logging.getLogger("go2_tide_bridge")


class Go2TideBridge:
    def __init__(self,
                 robot_id: str = "cash",
                 connection_method: str = "LocalSTA",
                 ip: Optional[str] = None,
                 serial: Optional[str] = None,
                 remote_user: Optional[str] = None,
                 remote_pass: Optional[str] = None,
                 cmd_watchdog_s: float = 0.6,
                 cmd_rate_hz: float = 20.0,
                 allow_lateral: bool = True):
        self.robot_id = robot_id
        self.cmd_watchdog_s = cmd_watchdog_s
        self.cmd_period = 1.0 / float(cmd_rate_hz)
        self.allow_lateral = allow_lateral

        # WebRTC connection settings
        self.conn_method = connection_method
        self.ip = ip
        self.serial = serial
        self.remote_user = remote_user
        self.remote_pass = remote_pass

        # Go2 connection object
        self.conn: Optional[Go2WebRTCConnection] = None

        # Zenoh session + pubs
        self.zenoh_session = None
        self.pose3d_pub = None
        self.quat_pub = None

        # Tide topic keys
        self.cmd_twist_key = robot_topic(self.robot_id, CmdTopic.TWIST.value).strip('/')
        self.pose3d_key = robot_topic(self.robot_id, StateTopic.POSE3D.value).strip('/')
        self.quat_key = robot_topic(self.robot_id, "sensor/imu/quat").strip('/')

        # Command state (from Tide)
        self._last_cmd_time = 0.0
        self._target_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # vx, vy, wz
        self._cmd_lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # -------------------- Go2 side --------------------
    async def _connect_go2(self):
        method = {
            "LocalAP": WebRTCConnectionMethod.LocalAP,
            "LocalSTA": WebRTCConnectionMethod.LocalSTA,
            "Remote": WebRTCConnectionMethod.Remote,
        }.get(self.conn_method, WebRTCConnectionMethod.LocalSTA)

        if method == WebRTCConnectionMethod.LocalSTA:
            if not self.ip and not self.serial:
                # Attempt auto-discovery if available
                if _discover_ip_sn is not None:
                    log.info("No IP/serial provided. Attempting Go2 discovery...")
                    try:
                        found = _discover_ip_sn(timeout=3)
                        if found:
                            # Pick first discovered device
                            sn, ip = next(iter(found.items()))
                            self.serial = sn
                            self.ip = ip
                            log.info(f"Discovered Go2 {sn} at {ip}")
                        else:
                            raise ValueError("Discovery found no devices. Provide GO2_IP or GO2_SERIAL.")
                    except Exception as e:
                        raise ValueError(f"Discovery failed: {e}. Provide GO2_IP or GO2_SERIAL.")
                else:
                    raise ValueError("LocalSTA requires either 'ip' or 'serial' to be provided")
            self.conn = Go2WebRTCConnection(method, ip=self.ip, serialNumber=self.serial)
        elif method == WebRTCConnectionMethod.Remote:
            if not self.serial or not self.remote_user or not self.remote_pass:
                raise ValueError("Remote requires 'serial', 'remote_user', and 'remote_pass'")
            self.conn = Go2WebRTCConnection(method, serialNumber=self.serial,
                                            username=self.remote_user, password=self.remote_pass)
        else:
            self.conn = Go2WebRTCConnection(method)

        await self.conn.connect()
        log.info("Connected to Go2 WebRTC service")

        # Subscribe to LF sport mode state for pose + IMU
        self.conn.datachannel.pub_sub.subscribe(
            RTC_TOPIC['LF_SPORT_MOD_STATE'], self._on_go2_state
        )

        # Ensure robot is in a movable mode and standing
        try:
            await self._ensure_motion_mode_normal()
            await asyncio.sleep(1.0)
            await self._stand_up()
            await asyncio.sleep(5.0)
            await self._set_speed_level(2)
        except Exception as e:
            log.warning(f"Motion mode/standup setup failed: {e}")

    async def _send_go2_move(self, vx: float, vy: float, wz: float):
        if not self.conn:
            return
        try:
            # Clamp to [-1.0, 1.0] – Unitree API typically expects normalized inputs
            def clamp(v, lo=-1.0, hi=1.0):
                return max(lo, min(hi, v))
            vx, vy, wz = vx, vy, wz

            resp = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {"x": float(vx), "y": float(vy), "z": float(wz)},
                },
            )
            code = None
            try:
                code = resp['data']['header']['status']['code']
            except Exception:
                pass
            if code == 0:
                log.debug(f"Sent Move OK: x={vx:.2f} y={vy:.2f} z={wz:.2f}")
            else:
                log.warning(f"Move rejected code={code}: x={vx:.2f} y={vy:.2f} z={wz:.2f}")
        except Exception as e:
            log.warning(f"Failed to send Move command: {e}")

    async def _send_go2_stop(self):
        # Send zero velocities to stop; StopMove also exists but zeroing is smooth
        await self._send_go2_move(0.0, 0.0, 0.0)

    def _on_go2_state(self, message: Dict[str, Any]):
        """Callback from Go2 driver when sport mode state arrives."""
        try:
            data = message.get('data', {})
            imu_state = data.get('imu_state', {})
            position = data.get('position', None)

            # Extract quaternion
            q = self._extract_quaternion(imu_state.get('quaternion'))
            if q is not None:
                qw, qx, qy, qz = q
                quat_msg = Quaternion(w=qw, x=qx, y=qy, z=qz)
                self.pose3d_pub and self.pose3d_pub.put(
                    to_zenoh_value(
                        Pose3D(position=Vector3(x=0.0, y=0.0, z=0.0),  # may be overwritten below
                               orientation=quat_msg)
                    )
                )
                self.quat_pub and self.quat_pub.put(to_zenoh_value(quat_msg))

            # Extract position (x,y,z)
            pos = self._extract_position(position)
            if pos is not None:
                px, py, pz = pos
                # If we already sent pose with quaternion above, we should resend including position
                # Publish Pose3D either way
                orientation = None
                # Best-effort: if last quaternion known via previous calls isn't cached, we just publish position only
                # Here, publish position with identity quaternion if none in this message
                if q is None:
                    orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
                else:
                    qw, qx, qy, qz = q
                    orientation = Quaternion(w=qw, x=qx, y=qy, z=qz)

                self.pose3d_pub and self.pose3d_pub.put(
                    to_zenoh_value(
                        Pose3D(position=Vector3(x=px, y=py, z=pz), orientation=orientation)
                    )
                )

        except Exception as e:
            log.debug(f"Error handling Go2 state: {e}")

    async def _ensure_motion_mode_normal(self):
        if not self.conn:
            return
        # Query current motion mode
        resp = await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
        )
        try:
            st = resp['data']['header']['status']['code']
            if st != 0:
                raise RuntimeError(f"Query motion mode failed code={st}")
            import json as _json
            data = _json.loads(resp['data']['data'])
            current = data.get('name')
        except Exception:
            current = None
        if current != 'normal':
            log.info(f"Switching motion mode to 'normal' (was {current})")
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1002, "parameter": {"name": "normal"}}
            )
            await asyncio.sleep(4.0)

    async def _stand_up(self):
        if not self.conn:
            return
        try:
            # Try BalanceStand first
            try:
                resp = await self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]}
                )
                code = resp['data']['header']['status']['code']
                log.info(f"BalanceStand code={code}")
            except Exception:
                pass

            await asyncio.sleep(0.2)
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}
            )
            log.info("Sent StandUp")
        except Exception as e:
            log.debug(f"StandUp failed: {e}")

    async def _set_speed_level(self, level: int):
        if not self.conn:
            return
        level = max(1, min(3, int(level)))
        try:
            resp = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["SpeedLevel"], "parameter": {"data": level}}
            )
            code = resp['data']['header']['status']['code']
            log.info(f"SpeedLevel {level} code={code}")
        except Exception as e:
            log.debug(f"SpeedLevel failed: {e}")

    

    @staticmethod
    def _extract_quaternion(raw: Any) -> Optional[Tuple[float, float, float, float]]:
        if raw is None:
            return None
        try:
            # Accept dict forms
            if isinstance(raw, dict):
                # Common variants
                keys = {k.lower(): k for k in raw.keys()}
                def getk(*cands):
                    for c in cands:
                        if c in keys:
                            return float(raw[keys[c]])
                    raise KeyError
                qw = getk('w', 'qw')
                qx = getk('x', 'qx')
                qy = getk('y', 'qy')
                qz = getk('z', 'qz')
                return (qw, qx, qy, qz)
            # Accept list/tuple [w,x,y,z] or [x,y,z,w]
            if isinstance(raw, (list, tuple)):
                if len(raw) == 4:
                    # Heuristic: Unitree tends to provide [w,x,y,z]
                    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        except Exception:
            return None
        return None

    @staticmethod
    def _extract_position(raw: Any) -> Optional[Tuple[float, float, float]]:
        if raw is None:
            return None
        try:
            if isinstance(raw, dict):
                keys = {k.lower(): k for k in raw.keys()}
                def getk(name):
                    return float(raw[keys[name]]) if name in keys else None
                x = getk('x')
                y = getk('y')
                z = getk('z')
                if x is None or y is None or z is None:
                    return None
                return (x, y, z)
            if isinstance(raw, (list, tuple)) and len(raw) >= 3:
                return (float(raw[0]), float(raw[1]), float(raw[2]))
        except Exception:
            return None
        return None

    # -------------------- Tide (Zenoh) side --------------------
    def _on_cmd_twist(self, sample):
        try:
            twist: Twist2D = from_zenoh_value(sample.payload, Twist2D)
        except Exception as e:
            log.warning(f"Failed to parse Twist2D from Tide cmd: {e}")
            return

        vx = float(twist.linear.x)
        vy = float(twist.linear.y)
        wz = float(twist.angular)

        # Update target from Zenoh callback thread safely
        with self._cmd_lock:
            self._target_cmd = (vx, vy, wz)
            self._last_cmd_time = time.monotonic()
        log.debug(f"Twist cmd: vx={vx:.2f} vy={vy:.2f} wz={wz:.2f}")

    async def _connect_zenoh(self):
        z_conf = zenoh.Config()
        self.zenoh_session = zenoh.open(z_conf)

        # Declare pubs
        self.pose3d_pub = self.zenoh_session.declare_publisher(self.pose3d_key)
        self.quat_pub = self.zenoh_session.declare_publisher(self.quat_key)

        # Subscribe to cmd/twist
        self.zenoh_session.declare_subscriber(self.cmd_twist_key, self._on_cmd_twist)

        log.info(f"Tide subscribe: {self.cmd_twist_key}")
        log.info(f"Tide publish pose3d: {self.pose3d_key}")
        log.info(f"Tide publish quat:   {self.quat_key}")

    # -------------------- Pump task --------------------
    async def _cmd_pump(self):
        """Periodically send the latest command to Go2, zeroing when stale."""
        try:
            last_sent: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            last_time = 0.0
            EPS = 0.03  # minimal change to resend
            REFRESH = 0.5  # seconds
            while True:
                now = time.monotonic()
                with self._cmd_lock:
                    vx, vy, wz = self._target_cmd
                stale = (now - self._last_cmd_time) > self.cmd_watchdog_s
                if stale:
                    vx = vy = wz = 0.0

                # Decide whether to send: if changed significantly or refresh interval hit
                dv = abs(vx - last_sent[0]) + abs(vy - last_sent[1]) + abs(wz - last_sent[2])
                need_refresh = (now - last_time) > REFRESH

                if dv > EPS or need_refresh:
                    if vx == 0.0 and vy == 0.0 and wz == 0.0:
                        # Prefer explicit StopMove when zero
                        try:
                            await self.conn.datachannel.pub_sub.publish_request_new(
                                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StopMove"]}
                            )
                            log.debug("Sent StopMove")
                        except Exception as e:
                            log.debug(f"StopMove failed: {e}")
                    else:
                        await self._send_go2_move(vx, vy, wz)
                    last_sent = (vx, vy, wz)
                    last_time = now
                await asyncio.sleep(self.cmd_period)
        except asyncio.CancelledError:
            pass

    async def run(self):
        # Cache running loop for any thread-safe interactions if needed
        self._loop = asyncio.get_running_loop()

        await self._connect_go2()
        await self._connect_zenoh()

        log.info("Bridge running. Listening for Tide cmd/twist and publishing Go2 state...")
        # Optional quick test move
        if os.environ.get("GO2_TEST_MOVE"):
            try:
                log.info("GO2_TEST_MOVE set: sending test forward 0.2 for 2s")
                await self._send_go2_move(0.2, 0.0, 0.0)
                await asyncio.sleep(2.0)
                await self._send_go2_move(0.0, 0.0, 0.0)
            except Exception as e:
                log.warning(f"GO2_TEST_MOVE failed: {e}")
        pump = asyncio.create_task(self._cmd_pump())
        try:
            # Run forever
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            pump.cancel()
            try:
                await pump
            except Exception:
                pass
            await self._shutdown()

    async def _shutdown(self):
        try:
            await self._send_go2_stop()
        except Exception:
            pass
        try:
            if self.zenoh_session:
                self.zenoh_session.close()
        except Exception:
            pass
        try:
            if self.conn:
                await asyncio.sleep(0.1)
                # No explicit close required in current driver API
        except Exception:
            pass
        log.info("Bridge stopped")


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


async def main():
    logging.basicConfig(level=os.environ.get("LOG", "INFO"))

    robot_id = _env("ROBOT_ID", "cash")

    # Choose connection details via env vars
    method = _env("GO2_METHOD", "LocalSTA")  # LocalAP | LocalSTA | Remote
    ip = _env("GO2_IP")
    serial = _env("GO2_SERIAL")
    user = _env("GO2_REMOTE_USER")
    pw = _env("GO2_REMOTE_PASS")

    bridge = Go2TideBridge(
        robot_id=robot_id,
        connection_method=method,
        ip=ip,
        serial=serial,
        remote_user=user,
        remote_pass=pw,
        cmd_watchdog_s=float(_env("CMD_WATCHDOG_S", "0.6")),
        cmd_rate_hz=float(_env("CMD_RATE_HZ", "20.0")),
        allow_lateral=_env("GO2_ALLOW_LATERAL", "false").lower() in ("1","true","yes"),
    )
    await bridge.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
