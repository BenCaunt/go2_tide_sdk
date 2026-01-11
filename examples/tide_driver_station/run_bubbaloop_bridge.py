#!/usr/bin/env python3
"""
Simple Bubbaloop Bridge - Bridges Go2 camera to Bubbaloop dashboard.

Run this in a separate terminal while Tide is running:
    python run_bubbaloop_bridge.py

Prerequisites:
    - pip install zenoh cbor2 opencv-python numpy
    - Tide running with Go2 connected
    - Zenoh bridge running: zenoh-bridge-remote-api --listen tcp/0.0.0.0:7448 --ws-port 10000
    - Bubbaloop dashboard: cd bubbaloop/dashboard && npm run dev
"""

import time
import signal
import sys
import numpy as np
import cv2
import zenoh
import cbor2

# Configuration
ROBOT_ID = "cash"
CAMERA_NAME = "go2_front"
ZENOH_ENDPOINT = "tcp/127.0.0.1:7448"
JPEG_QUALITY = 80
TARGET_SIZE = (640, 480)

# State
running = True
frame_count = 0


def encode_varint(value):
    result = []
    while value > 127:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def encode_protobuf_message(jpeg_data, sequence):
    """Encode CompressedImage protobuf message."""
    now_ns = int(time.time() * 1e9)

    # Header
    header = b''
    header += encode_varint((1 << 3) | 0) + encode_varint(now_ns)  # acq_time
    header += encode_varint((2 << 3) | 0) + encode_varint(now_ns)  # pub_time
    header += encode_varint((3 << 3) | 0) + encode_varint(sequence)  # sequence
    frame_id = CAMERA_NAME.encode()
    header += encode_varint((4 << 3) | 2) + encode_varint(len(frame_id)) + frame_id  # frame_id

    # CompressedImage
    msg = b''
    msg += encode_varint((1 << 3) | 2) + encode_varint(len(header)) + header  # header
    fmt = b"jpeg"
    msg += encode_varint((2 << 3) | 2) + encode_varint(len(fmt)) + fmt  # format
    msg += encode_varint((3 << 3) | 2) + encode_varint(len(jpeg_data)) + jpeg_data  # data

    return msg


def on_image(sample):
    global frame_count

    try:
        data = sample.payload.to_bytes()

        # Parse CBOR
        msg = cbor2.loads(data)
        height = msg.get("height", 0)
        width = msg.get("width", 0)
        encoding = msg.get("encoding", "bgr8")
        img_data = msg.get("data")

        if not img_data or height == 0 or width == 0:
            return

        # Convert to numpy
        img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))

        # Convert RGB to BGR if needed
        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize
        img = cv2.resize(img, TARGET_SIZE)

        # Encode as JPEG
        success, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not success:
            return

        jpeg_bytes = jpeg.tobytes()

        # Encode and publish protobuf message
        proto_msg = encode_protobuf_message(jpeg_bytes, frame_count)
        publisher.put(proto_msg)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Published {frame_count} frames ({len(jpeg_bytes)} bytes/frame)")

    except Exception as e:
        print(f"Error: {e}")


def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 50)
    print("Bubbaloop Bridge for Go2")
    print("=" * 50)
    print(f"Robot ID: {ROBOT_ID}")
    print(f"Camera name: {CAMERA_NAME}")
    print(f"Zenoh endpoint: {ZENOH_ENDPOINT}")
    print()

    # Connect to Zenoh
    print("Connecting to Zenoh...")
    config = zenoh.Config()
    config.insert_json5("connect/endpoints", f'["{ZENOH_ENDPOINT}"]')
    session = zenoh.open(config)
    print("Connected!")

    # Setup publisher and subscriber
    input_topic = f"{ROBOT_ID}/sensor/camera/front/image"
    output_topic = f"camera/{CAMERA_NAME}/compressed"

    print(f"Subscribing to: {input_topic}")
    print(f"Publishing to: {output_topic}")

    publisher = session.declare_publisher(output_topic)
    subscriber = session.declare_subscriber(input_topic, on_image)

    print()
    print("Bridge running! Press Ctrl+C to stop.")
    print("Open Bubbaloop dashboard at https://localhost:5174")
    print(f"Add camera with topic: {output_topic}")
    print()

    # Main loop
    while running:
        time.sleep(0.1)

    # Cleanup
    subscriber.undeclare()
    publisher.undeclare()
    session.close()

    print(f"Stopped. Published {frame_count} total frames.")
