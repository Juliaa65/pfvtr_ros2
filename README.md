# PFVTR ROS2

## Overview

PFVTR is a ROS 2 visual teach-and-repeat system

# Build

From the workspace root:

```bash
colcon build
```

---

# Source Environment

Example for ROS 2 Kilted:

```bash
source /opt/ros/kilted/setup.bash
source install/setup.bash
```
# Running PFVTR

## Launch

```bash
ros2 launch pfvtr sim.launch.py
```

---

# Robot-Specific Topics

These are the main topics that usually need to be changed depending on the robot platform

| Topic | Description |
|---|---|
| `camera_topic` | Main front camera image topic |
| `odom_topic` | Main robot odometry topic |
| `cmd_vel_robot_input` | Velocity command topic used to control the robot base |

---

# Example Launch

```bash
ros2 launch pfvtr sim.launch.py \
  camera_topic:=/camera/image_raw \
  odom_topic:=/odom \
  cmd_vel_robot_input:=/cmd_vel
```

---

# Teach Phase

Start map recording:

```bash
ros2 action send_goal \
  /pfvtr/mapmaker \
  pfvtr/action/MapMaker \
  "{
    save_imgs_for_viz: true,
    map_name: 'my_map',
    start: true,
    map_step: 1.0,
    source_map: '',
    record_backward: false
  }"
```

Drive the robot manually while recording.

---

# Repeat Phase

Start route repeat:

```bash
ros2 action send_goal \
  /pfvtr/repeater \
  pfvtr/action/MapRepeater \
  "{
    start_pos: 0.0,
    end_pos: 0.0,
    traversals: 1,
    null_cmd: false,
    image_pub: 1,
    use_dist: true,
    map_name: 'my_map'
  }"
```

The robot will autonomously repeat the recorded route.