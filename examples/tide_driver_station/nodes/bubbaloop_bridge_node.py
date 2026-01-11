"""
Bubbaloop Bridge Node - Publishes Go2 camera frames to Bubbaloop dashboard via Zenoh.

This node bridges the Go2's WebRTC camera stream to Bubbaloop's dashboard by:
1. Receiving camera images from the Go2 via Tide topics
2. Encoding them as JPEG
3. Publishing them in Bubbaloop's protobuf format via Zenoh

Prerequisites:
- pip install zenoh opencv-python
- Bubbaloop dashboard running (npm run dev in bubbaloop/dashboard)
- Zenoh bridge running (zenoh-bridge-remote-api)
"""

import time
import logging
from typing import Any, Dict, Optional
import numpy as np
import cv2

try:
    import zenoh
except ImportError:
    zenoh = None
    logging.warning("zenoh not installed. Run: pip install zenoh")

from tide.core.node import BaseNode


class BubballoopBridgeNode(BaseNode):
    """Bridge Go2 camera to Bubbaloop dashboard via Zenoh."""

    GROUP = ""

    def __init__(self, *, config=None):
        super().__init__(config=config)
        p = config or {}

        self.hz = float(p.get("update_rate", 30.0))
        self.camera_name = p.get("camera_name", "go2_front")
        self.image_topic = p.get("image_topic", "sensor/camera/front/image")
        self.zenoh_endpoint = p.get("zenoh_endpoint", "tcp/127.0.0.1:7448")
        self.jpeg_quality = int(p.get("jpeg_quality", 80))
        self.target_width = int(p.get("target_width", 640))
        self.target_height = int(p.get("target_height", 480))

        self.zenoh_session: Optional[Any] = None
        self.publisher = None
        self.sequence = 0
        self._last_img: Optional[Dict[str, Any]] = None

        # Subscribe to camera images
        self.subscribe(self.image_topic, self._on_image)

        # Initialize Zenoh connection
        self._init_zenoh()

        logging.info(f"[BubballoopBridge] Camera: {self.camera_name}, Topic: {self.image_topic}")

    def _init_zenoh(self):
        """Initialize Zenoh connection."""
        if zenoh is None:
            logging.error("[BubballoopBridge] zenoh package not available. Run: pip install zenoh")
            return

        try:
            # Configure Zenoh to connect to the bridge
            config = zenoh.Config()
            config.insert_json5("connect/endpoints", f'["{self.zenoh_endpoint}"]')

            self.zenoh_session = zenoh.open(config)

            # Declare publisher for compressed images
            # Note: Zenoh key expressions must NOT have leading slashes
            topic = f"camera/{self.camera_name}/compressed"
            self.publisher = self.zenoh_session.declare_publisher(topic)

            logging.info(f"[BubballoopBridge] Connected to Zenoh, publishing to {topic}")
        except Exception as e:
            logging.error(f"[BubballoopBridge] Failed to connect to Zenoh: {e}")
            self.zenoh_session = None

    def _on_image(self, msg: Dict[str, Any]):
        """Cache latest camera image."""
        self._last_img = msg

    def _encode_compressed_image(self, jpeg_data: bytes, acq_time_ns: int) -> bytes:
        """
        Encode a CompressedImage protobuf message.

        Bubbaloop proto format:
        message Header {
            uint64 acq_time = 1;
            uint64 pub_time = 2;
            uint32 sequence = 3;
            string frame_id = 4;
        }
        message CompressedImage {
            Header header = 1;
            string format = 2;
            bytes data = 3;
        }
        """
        pub_time_ns = int(time.time() * 1e9)

        # Build Header (field 1 in CompressedImage)
        header_bytes = b''
        # acq_time (field 1, varint)
        header_bytes += self._encode_varint_field(1, acq_time_ns)
        # pub_time (field 2, varint)
        header_bytes += self._encode_varint_field(2, pub_time_ns)
        # sequence (field 3, varint)
        header_bytes += self._encode_varint_field(3, self.sequence)
        # frame_id (field 4, string)
        header_bytes += self._encode_string_field(4, self.camera_name)

        # Build CompressedImage
        msg_bytes = b''
        # header (field 1, length-delimited)
        msg_bytes += self._encode_bytes_field(1, header_bytes)
        # format (field 2, string)
        msg_bytes += self._encode_string_field(2, "jpeg")
        # data (field 3, bytes)
        msg_bytes += self._encode_bytes_field(3, jpeg_data)

        return msg_bytes

    def _encode_varint(self, value: int) -> bytes:
        """Encode an integer as a protobuf varint."""
        result = []
        while value > 127:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value)
        return bytes(result)

    def _encode_varint_field(self, field_num: int, value: int) -> bytes:
        """Encode a varint field with its tag."""
        tag = (field_num << 3) | 0  # wire type 0 = varint
        return self._encode_varint(tag) + self._encode_varint(value)

    def _encode_string_field(self, field_num: int, value: str) -> bytes:
        """Encode a string field with its tag."""
        tag = (field_num << 3) | 2  # wire type 2 = length-delimited
        value_bytes = value.encode('utf-8')
        return self._encode_varint(tag) + self._encode_varint(len(value_bytes)) + value_bytes

    def _encode_bytes_field(self, field_num: int, value: bytes) -> bytes:
        """Encode a bytes field with its tag."""
        tag = (field_num << 3) | 2  # wire type 2 = length-delimited
        return self._encode_varint(tag) + self._encode_varint(len(value)) + value

    def step(self):
        """Process and publish camera frames to Bubbaloop."""
        if self.zenoh_session is None or self.publisher is None:
            return

        if self._last_img is None:
            return

        try:
            msg = self._last_img
            height = msg.get("height", 0)
            width = msg.get("width", 0)
            encoding = msg.get("encoding", "bgr8")
            data = msg.get("data")

            if data is None or height == 0 or width == 0:
                return

            # Convert to numpy array
            if isinstance(data, bytes):
                if encoding in ("bgr8", "rgb8"):
                    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                else:
                    return
            elif isinstance(data, np.ndarray):
                img = data
            else:
                return

            # Convert RGB to BGR for OpenCV if needed
            if encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Resize if needed
            if self.target_width > 0 and self.target_height > 0:
                if img.shape[1] != self.target_width or img.shape[0] != self.target_height:
                    img = cv2.resize(img, (self.target_width, self.target_height))

            # Encode as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            success, jpeg_data = cv2.imencode('.jpg', img, encode_params)

            if not success:
                return

            jpeg_bytes = jpeg_data.tobytes()

            # Build protobuf message
            acq_time_ns = int(time.time() * 1e9)
            msg_bytes = self._encode_compressed_image(jpeg_bytes, acq_time_ns)

            # Publish via Zenoh
            self.publisher.put(msg_bytes)

            self.sequence += 1

            if self.sequence % 100 == 0:
                logging.info(f"[BubballoopBridge] Published {self.sequence} frames")

        except Exception as e:
            logging.error(f"[BubballoopBridge] Error processing frame: {e}")

    def cleanup(self):
        """Clean up Zenoh connection."""
        if self.publisher:
            try:
                self.publisher.undeclare()
            except Exception:
                pass
        if self.zenoh_session:
            try:
                self.zenoh_session.close()
            except Exception:
                pass
