#!/usr/bin/env bash
# One-time setup: install arduino-cli locally + ESP32 core,
# and install esptool on the Pi.
set -euo pipefail
cd "$(dirname "$0")"

source lib.sh
load_config

echo "=== Local setup ==="

if ! command -v arduino-cli &>/dev/null; then
  echo "Installing arduino-cli..."
  curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR="$HOME/.local/bin" sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "arduino-cli $(arduino-cli version)"

echo "Installing ESP32 board core (this takes a few minutes)..."
arduino-cli config init --overwrite 2>/dev/null || true
arduino-cli config add board_manager.additional_urls \
  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32

echo ""
echo "=== Pi setup ($PI_USER@$PI_HOST) ==="

ssh "$PI_USER@$PI_HOST" bash <<'REMOTE'
  if ! command -v esptool.py &>/dev/null; then
    echo "Installing esptool..."
    pip3 install --user esptool
  fi
  echo "esptool: $(esptool.py version 2>&1 | head -1)"

  # Ensure user can access serial ports
  if ! groups | grep -q dialout; then
    echo "Adding $USER to dialout group (re-login needed)..."
    sudo usermod -aG dialout "$USER"
  fi

  echo "Serial devices:"
  ls -la /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  (none found — is the ESP32 plugged in?)"
REMOTE

echo ""
echo "=== Setup complete ==="
echo "Run: ./deploy.sh   (build + flash)"
echo "Run: ./monitor.sh  (serial output)"
