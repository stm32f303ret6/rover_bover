#!/usr/bin/env bash
# Compile the ESP32 sketch locally using arduino-cli
set -euo pipefail
cd "$(dirname "$0")"

source lib.sh
load_config

echo "==> Compiling sketch: $SKETCH_DIR"
mkdir -p "$BUILD_DIR"

arduino-cli compile \
  --fqbn "$FQBN" \
  --output-dir "$BUILD_DIR" \
  "$SKETCH_DIR"

echo "==> Build complete: $BUILD_DIR"
ls -lh "$BUILD_DIR"/*.bin 2>/dev/null
