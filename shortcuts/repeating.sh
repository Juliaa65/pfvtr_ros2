#!/usr/bin/env bash
# Start/stop a pfvtr repeat (navigate) pass via /pfvtr/repeater. Parameters
# default from repeating.cfg (see ../action/MapRepeater.action).
#
# `start` blocks until the repeat completes or is cancelled — run `stop`
# from a second terminal to cancel it early via the /pfvtr/stop_repeater
# service (Ctrl+C in the first terminal also cancels the goal client-side).
#
# Assumes your ROS2 environment is already sourced.
#
# Usage:
#   ./repeating.sh start [map_name]
#   ./repeating.sh stop
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source <(python3 "${SCRIPT_DIR}/yaml_to_env.py" "${SCRIPT_DIR}/repeating.cfg")

usage() { echo "usage: $0 {start|stop} [map_name]" >&2; exit 1; }

[[ $# -ge 1 ]] || usage
CMD="$1"; shift

case "$CMD" in
  start)
    NAME="${1:-$MAP_NAME}"
    echo "Repeating: /pfvtr/repeater  map_name=${NAME}"
    ros2 action send_goal /pfvtr/repeater pfvtr/action/MapRepeater \
      "{start_pos: ${START_POS}, end_pos: ${END_POS}, null_cmd: ${NULL_CMD}, image_pub: ${IMAGE_PUB}, map_name: '${NAME}', publish_trajectory: ${PUBLISH_TRAJECTORY}, use_cmd_vel: ${USE_CMD_VEL}, trajectory_horizon: ${TRAJECTORY_HORIZON}}"
    ;;
  stop)
    echo "Stopping repeat: /pfvtr/stop_repeater"
    ros2 service call /pfvtr/stop_repeater pfvtr/srv/StopRepeater "{sto: true}"
    ;;
  *)
    usage
    ;;
esac
