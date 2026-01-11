#!/bin/bash
# Bubbaloop Integration Setup for Go2
# This script helps set up the Bubbaloop dashboard to visualize Go2 camera feeds

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUBBALOOP_DIR="$(cd "$SCRIPT_DIR/../../bubbaloop" && pwd)"

echo "=== Bubbaloop + Go2 Integration Setup ==="
echo ""

# Check if bubbaloop exists
if [ ! -d "$BUBBALOOP_DIR" ]; then
    echo "Error: Bubbaloop not found at $BUBBALOOP_DIR"
    echo "Please run: git clone https://github.com/kornia/bubbaloop.git $BUBBALOOP_DIR"
    exit 1
fi

# Install Python dependencies
echo "1. Installing Python dependencies..."
pip install zenoh opencv-python-headless 2>/dev/null || pip install zenoh opencv-python

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "Error: npm is required. Please install Node.js first."
    exit 1
fi

# Install dashboard dependencies
echo "2. Installing dashboard dependencies..."
cd "$BUBBALOOP_DIR/dashboard"
npm install

# Build zenoh bridge
echo "3. Setting up Zenoh bridge..."
cd "$BUBBALOOP_DIR"
if [ ! -d "zenoh-ts" ]; then
    git clone https://github.com/eclipse-zenoh/zenoh-ts.git
fi

# Check for Rust
if command -v cargo &> /dev/null; then
    echo "   Building zenoh-bridge-remote-api..."
    cd zenoh-ts/zenoh-bridge-remote-api
    cargo build --release 2>/dev/null || echo "   Note: Rust build failed, you may need to install the bridge manually"
else
    echo "   Warning: Rust/cargo not found. Install it to build the Zenoh bridge."
    echo "   Alternatively, download a pre-built binary from:"
    echo "   https://github.com/eclipse-zenoh/zenoh-ts/releases"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the Bubbaloop dashboard with Go2:"
echo ""
echo "Terminal 1 - Start Zenoh bridge:"
echo "  cd $BUBBALOOP_DIR"
echo "  ./zenoh-ts/target/release/zenoh-bridge-remote-api --listen tcp/0.0.0.0:7448 --ws-port 10000"
echo ""
echo "Terminal 2 - Start dashboard:"
echo "  cd $BUBBALOOP_DIR/dashboard"
echo "  npm run dev"
echo ""
echo "Terminal 3 - Start Go2 driver station with Bubbaloop bridge:"
echo "  cd $SCRIPT_DIR"
echo "  # Edit config/config.yaml to add BubballoopBridgeNode"
echo "  tide run config/config.yaml"
echo ""
echo "Then open http://localhost:5173 in your browser"
echo "Click 'Add Camera' and enter topic: /camera/go2_front/compressed"
