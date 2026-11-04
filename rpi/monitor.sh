#!/usr/bin/env bash
# Open serial monitor on the Pi (Ctrl+C to exit)
set -euo pipefail
cd "$(dirname "$0")"

source lib.sh
load_config

echo "==> Serial monitor on $PI_HOST:$ESP_PORT at 115200 baud (Ctrl+C to exit)"
ssh "$PI_USER@$PI_HOST" "cat $ESP_PORT" || \
ssh "$PI_USER@$PI_HOST" "python3 -c \"
import serial, sys
s = serial.Serial('$ESP_PORT', 115200, timeout=1)
try:
  while True:
    line = s.readline()
    if line:
      sys.stdout.buffer.write(line)
      sys.stdout.flush()
except KeyboardInterrupt:
  pass
\""
