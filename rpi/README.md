# ESP32 Flash Pipeline

Compile Arduino sketches locally and flash an ESP32 connected to a Raspberry Pi over SSH.

## Setup

```
Pi:   rsc@10.0.0.1 (AP on wlan1)
ESP32: ESP32-D0WD-V3, CP2102 on /dev/ttyUSB0
```

### One-time setup

**Local (already done):**
```bash
# arduino-cli + ESP32 core
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR=~/.local/bin sh
arduino-cli config add board_manager.additional_urls https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index && arduino-cli core install esp32:esp32
```

**Pi (already done):**
```bash
python3 -m venv ~/esp-env
~/esp-env/bin/pip install esptool
```

## Usage

```bash
./deploy.sh          # build + flash (most common)
./build.sh           # compile only
./flash.sh           # flash only (app at 0x10000)
./flash.sh --full    # flash bootloader + partitions + app
./monitor.sh         # serial monitor over SSH (Ctrl+C to exit)
```

## Configuration

Edit `config.yaml` to change Pi host, serial port, baud rate, or board FQBN.

## Project structure

```
rpi/
├── config.yaml       # Pi/ESP32/build settings
├── sketch/sketch.ino # Arduino sketch
├── lib.sh            # Shared config parser
├── build.sh          # Local compile
├── flash.sh          # SCP + esptool flash
├── deploy.sh         # build + flash
├── monitor.sh        # Serial monitor over SSH
└── setup.sh          # One-time install script
```
