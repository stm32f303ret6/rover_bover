#!/usr/bin/env bash
# Shared config loader — sourced by other scripts, not run directly.
# Parses config.yaml without requiring yq (pure bash + grep/sed).

load_config() {
  local cfg
  cfg="$(dirname "${BASH_SOURCE[0]}")/config.yaml"

  if [[ ! -f "$cfg" ]]; then
    echo "ERROR: config.yaml not found" >&2
    exit 1
  fi

  yaml_val() { grep "^  $1:" "$cfg" | sed "s/^  $1: *//" | sed 's/ *#.*//' ; }

  PI_HOST="$(yaml_val host)"
  PI_USER="$(yaml_val user)"
  PI_DEPLOY_DIR="$(yaml_val deploy_dir)"
  FQBN="$(yaml_val fqbn)"
  ESP_PORT="$(yaml_val port)"
  BAUD="$(yaml_val baud)"
  SKETCH_DIR="$(yaml_val sketch_dir)"
  BUILD_DIR="$(yaml_val build_dir)"
}
