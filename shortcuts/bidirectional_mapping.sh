#!/usr/bin/env bash
# Start/stop simultaneous forward + backward pfvtr mapping (teach) sessions.
# Requires BOTH namespaces already running, e.g. via
# ../launch/dual_map.launch.py (or bringup's pfvtr.launch.py +
# pfvtr_map.launch.py).
#
# Forward map:  /pfvtr/mapmaker      <map_name>       (front camera)
# Backward map: /pfvtr_map/mapmaker  <map_name>_back  (rear camera, record_backward=true)
#
# Assumes your ROS2 environment is already sourced.
#
# Usage:
#   ./bidirectional_mapping.sh start [map_name]
#   ./bidirectional_mapping.sh stop  [map_name]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source <(python3 "${SCRIPT_DIR}/yaml_to_env.py" "${SCRIPT_DIR}/mapping.cfg")

usage() { echo "usage: $0 {start|stop} [map_name]" >&2; exit 1; }

[[ $# -ge 1 ]] || usage
CMD="$1"; shift
FWD_NAME="${1:-$MAP_NAME}"
BACK_NAME="${FWD_NAME}_back"

case "$CMD" in
  start)
    echo "Starting mapping: /pfvtr/mapmaker      map_name=${FWD_NAME}"
    ros2 action send_goal /pfvtr/mapmaker pfvtr/action/MapMaker \
      "{save_imgs_for_viz: ${SAVE_IMGS_FOR_VIZ}, map_name: '${FWD_NAME}', start: true, map_step: ${MAP_STEP}, source_map: '${SOURCE_MAP}', record_backward: false}"
    echo "Starting mapping: /pfvtr_map/mapmaker  map_name=${BACK_NAME}"
    ros2 action send_goal /pfvtr_map/mapmaker pfvtr/action/MapMaker \
      "{save_imgs_for_viz: ${SAVE_IMGS_FOR_VIZ}, map_name: '${BACK_NAME}', start: true, map_step: ${MAP_STEP}, source_map: '${SOURCE_MAP}', record_backward: true}"
    ;;
  stop)
    echo "Stopping mapping: /pfvtr/mapmaker      map_name=${FWD_NAME}"
    ros2 action send_goal /pfvtr/mapmaker pfvtr/action/MapMaker \
      "{map_name: '${FWD_NAME}', start: false}"
    echo "Stopping mapping: /pfvtr_map/mapmaker  map_name=${BACK_NAME}"
    ros2 action send_goal /pfvtr_map/mapmaker pfvtr/action/MapMaker \
      "{map_name: '${BACK_NAME}', start: false}"
    ;;
  *)
    usage
    ;;
esac
