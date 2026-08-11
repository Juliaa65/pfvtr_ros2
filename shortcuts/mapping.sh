#!/usr/bin/env bash
# Start/stop a pfvtr mapping (teach) session on the front camera, via
# /pfvtr/mapmaker. Parameters default from mapping.cfg (see ../action/MapMaker.action).
#
# Assumes your ROS2 environment is already sourced.
#
# Usage:
#   ./mapping.sh start [map_name]
#   ./mapping.sh stop  [map_name]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source <(python3 "${SCRIPT_DIR}/yaml_to_env.py" "${SCRIPT_DIR}/mapping.cfg")

usage() { echo "usage: $0 {start|stop} [map_name]" >&2; exit 1; }

[[ $# -ge 1 ]] || usage
CMD="$1"; shift
NAME="${1:-$MAP_NAME}"

case "$CMD" in
  start)
    echo "Starting mapping: /pfvtr/mapmaker  map_name=${NAME}"
    ros2 action send_goal /pfvtr/mapmaker pfvtr/action/MapMaker \
      "{save_imgs_for_viz: ${SAVE_IMGS_FOR_VIZ}, map_name: '${NAME}', start: true, map_step: ${MAP_STEP}, source_map: '${SOURCE_MAP}', record_backward: false}"
    ;;
  stop)
    echo "Stopping mapping: /pfvtr/mapmaker  map_name=${NAME}"
    ros2 action send_goal /pfvtr/mapmaker pfvtr/action/MapMaker \
      "{map_name: '${NAME}', start: false}"
    ;;
  *)
    usage
    ;;
esac
