# pfvtr_ros2

---

## Overview
This workspace is a ROS 2 port of the original PFVTR implementation. The main items ported to ROS 2 (and their ROS 2 equivalents) include:

- Nodes: the main runtime nodes were ported to ROS 2 (under `src/pfvtr/src/`): `controller`, `mapmaker` / `repeater`, and `sensors`.
- Interfaces: `msg/`, `srv/`, and `action/`.
- ROS 2 Python launch files `launch/sim.launch.py`)
---

## Prerequisites

- Ubuntu 24.04
- ROS 2 Jazzy
- Python (3.x)
---

## Build and Run

```bash

# Build the workspace
colcon build --symlink-install

# Source the local setup
source install/setup.bash

# After sourcing the workspace's install/setup.bash
ros2 launch pfvtr sim.launch.py
```
