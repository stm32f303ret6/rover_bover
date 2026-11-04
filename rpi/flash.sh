#!/usr/bin/env bash
# Send compiled binary to Pi and flash the ESP32
# Usage: ./flash.sh         (app only, fast)
#        ./flash.sh --full  (bootloader + partitions + app)
set -euo pipefail
cd "$(dirname "$0")"

source lib.sh
load_config

REMOTE="$PI_USER@$PI_HOST"

if [[ "${1:-}" == "--full" ]]; then
  for f in sketch.ino.bootloader.bin sketch.ino.partitions.bin sketch.ino.bin; do
    [[ -f "$BUILD_DIR/$f" ]] || { echo "ERROR: $BUILD_DIR/$f not found — run ./build.sh first"; exit 1; }
  done

  echo "==> Full flash: uploading bootloader + partitions + app"
  ssh "$REMOTE" "mkdir -p $PI_DEPLOY_DIR"
  scp "$BUILD_DIR"/sketch.ino.bootloader.bin "$BUILD_DIR"/sketch.ino.partitions.bin "$BUILD_DIR"/sketch.ino.bin "$REMOTE:$PI_DEPLOY_DIR/"

  echo "==> Flashing ESP32 on $ESP_PORT at $BAUD baud"
  ssh "$REMOTE" "~/esp-env/bin/esptool --chip esp32 --port $ESP_PORT --baud $BAUD write_flash \
    0x1000  $PI_DEPLOY_DIR/sketch.ino.bootloader.bin \
    0x8000  $PI_DEPLOY_DIR/sketch.ino.partitions.bin \
    0x10000 $PI_DEPLOY_DIR/sketch.ino.bin"
else
  BIN="$BUILD_DIR/sketch.ino.bin"
  [[ -f "$BIN" ]] || { echo "ERROR: No binary found at $BIN — run ./build.sh first"; exit 1; }

  echo "==> Uploading app binary to $REMOTE:$PI_DEPLOY_DIR"
  ssh "$REMOTE" "mkdir -p $PI_DEPLOY_DIR"
  scp "$BIN" "$REMOTE:$PI_DEPLOY_DIR/firmware.bin"

  echo "==> Flashing ESP32 on $ESP_PORT at $BAUD baud"
  ssh "$REMOTE" "~/esp-env/bin/esptool --chip esp32 --port $ESP_PORT --baud $BAUD write_flash 0x10000 $PI_DEPLOY_DIR/firmware.bin"
fi

echo "==> Flash complete"
