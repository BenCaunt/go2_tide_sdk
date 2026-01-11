#!/usr/bin/env python3
"""
Standalone Bubbaloop Bridge - Runs independently of Tide to bridge Go2 camera to Bubbaloop.

This script:
1. Subscribes to raw camera images from the Go2 via Zenoh
2. Encodes them as JPEG
3. Publishes to Bubbaloop's protobuf format

Usage:
    python bubbaloop_bridge.py [--robot-id cash] [--endpoint tcp/127.0.0.1:7448]
"""

import argparse
import time
import signal
import sys
import numpy as np
import cv2
import zenoh


class BubballoopBridge:
    def __init__(self, robot_id: str, endpoint: str, camera_name: str = "go2_front",
                 jpeg_quality: int = 80, target_size: tuple = (640, 480)):
        self.robot_id = robot_id
        self.camera_name = camera_name
        self.jpeg_quality = jpeg_quality
        self.target_width, self.target_height = target_size
        self.sequence = 0
        self.running = True

        # Connect to Zenoh
        print(f"Connecting to Zenoh at {endpoint}...")
        config = zenoh.Config()
        config.insert_json5("connect/endpoints", f'["{endpoint}"]')
        self.session = zenoh.open(config)
        print("Connected!")

        # Subscribe to raw camera topic
        self.input_topic = f"{robot_id}/sensor/camera/front/image"
        print(f"Subscribing to: {self.input_topic}")
        self.subscriber = self.session.declare_subscriber(self.input_topic, self._on_image)

        # Publish to Bubbaloop format
        self.output_topic = f"camera/{camera_name}/compressed"
        print(f"Publishing to: {self.output_topic}")
        self.publisher = self.session.declare_publisher(self.output_topic)

        print(f"Bridge running! JPEG quality={jpeg_quality}, target={target_size}")

    def _on_image(self, sample):
        """Process incoming camera frame and publish as JPEG."""
        try:
            data = sample.payload.to_bytes()

            # Parse the Tide message format (CBOR)
            try:
                import cbor2
                msg = cbor2.loads(data)
                height = msg.get("height", 0)
                width = msg.get("width", 0)
                encoding = msg.get("encoding", "bgr8")
                img_data = msg.get("data")
            except Exception:
                # Fallback to JSON
                try:
                    import json
                    msg = json.loads(data)
                    height = msg.get("height", 0)
                    width = msg.get("width", 0)
                    encoding = msg.get("encoding", "bgr8")
                    img_data = msg.get("data")
                    if isinstance(img_data, str):
                        import base64
                        img_data = base64.b64decode(img_data)
                except Exception:
                    return

            if not img_data or height == 0 or width == 0:
                return

            # Convert to numpy array
            if isinstance(img_data, bytes):
                img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
            else:
                return

            # Convert RGB to BGR if needed
            if encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Resize
            if self.target_width > 0 and self.target_height > 0:
                img = cv2.resize(img, (self.target_width, self.target_height))

            # Encode as JPEG
            success, jpeg_data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not success:
                return

            jpeg_bytes = jpeg_data.tobytes()

            # Build protobuf message
            acq_time_ns = int(time.time() * 1e9)
            msg_bytes = self._encode_compressed_image(jpeg_bytes, acq_time_ns)

            # Publish
            self.publisher.put(msg_bytes)
            self.sequence += 1

            if self.sequence % 30 == 0:
                print(f"Published {self.sequence} frames ({len(jpeg_bytes)} bytes/frame)")

        except Exception as e:
            print(f"Error processing frame: {e}")

    def _encode_compressed_image(self, jpeg_data: bytes, acq_time_ns: int) -> bytes:
        """Encode a CompressedImage protobuf message."""
        pub_time_ns = int(time.time() * 1e9)

        # Build Header
        header_bytes = b''
        header_bytes += self._encode_varint_field(1, acq_time_ns)
        header_bytes += self._encode_varint_field(2, pub_time_ns)
        header_bytes += self._encode_varint_field(3, self.sequence)
        header_bytes += self._encode_string_field(4, self.camera_name)

        # Build CompressedImage
        msg_bytes = b''
        msg_bytes += self._encode_bytes_field(1, header_bytes)
        msg_bytes += self._encode_string_field(2, "jpeg")
        msg_bytes += self._encode_bytes_field(3, jpeg_data)

        return msg_bytes

    def _encode_varint(self, value: int) -> bytes:
        result = []
        while value > 127:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value)
        return bytes(result)

    def _encode_varint_field(self, field_num: int, value: int) -> bytes:
        tag = (field_num << 3) | 0
        return self._encode_varint(tag) + self._encode_varint(value)

    def _encode_string_field(self, field_num: int, value: str) -> bytes:
        tag = (field_num << 3) | 2
        value_bytes = value.encode('utf-8')
        return self._encode_varint(tag) + self._encode_varint(len(value_bytes)) + value_bytes

    def _encode_bytes_field(self, field_num: int, value: bytes) -> bytes:
        tag = (field_num << 3) | 2
        return self._encode_varint(tag) + self._encode_varint(len(value)) + value

    def run(self):
        """Run the bridge until interrupted."""
        print("\nBridge running. Press Ctrl+C to stop.\n")

        def signal_handler(sig, frame):
            print("\nShutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        while self.running:
            time.sleep(0.1)

        self.subscriber.undeclare()
        self.publisher.undeclare()
        self.session.close()
        print("Bridge stopped.")


def main():
    parser = argparse.ArgumentParser(description="Bridge Go2 camera to Bubbaloop dashboard")
    parser.add_argument("--robot-id", default="cash", help="Robot ID (default: cash)")
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7448", help="Zenoh endpoint")
    parser.add_argument("--camera-name", default="go2_front", help="Camera name for Bubbaloop")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (1-100)")
    parser.add_argument("--width", type=int, default=640, help="Target width")
    parser.add_argument("--height", type=int, default=480, help="Target height")

    args = parser.parse_args()

    bridge = BubballoopBridge(
        robot_id=args.robot_id,
        endpoint=args.endpoint,
        camera_name=args.camera_name,
        jpeg_quality=args.quality,
        target_size=(args.width, args.height)
    )

    bridge.run()


if __name__ == "__main__":
    main()
