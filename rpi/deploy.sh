#!/usr/bin/env bash
# One-command build + flash pipeline
set -euo pipefail
cd "$(dirname "$0")"

./build.sh
./flash.sh
