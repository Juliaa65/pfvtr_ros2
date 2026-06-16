#!/usr/bin/env python3
"""Publish a live plot of /pfvtr/particles as sensor_msgs/Image.

Subscribes to /pfvtr/particles (published by /pfvtr/sensors when debug=true).
Optionally toggles sensors debug via manage_sensors_debug (off by default).
"""

import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from pfvtr.msg import FloatList

SET_PARAM_SERVICE = "/pfvtr/sensors/set_parameters"


class ParticlesViewer(Node):
    def __init__(self):
        super().__init__("particles_viewer")
        self.declare_parameter("particles_topic", "/pfvtr/particles")
        self.declare_parameter("image_topic", "/pfvtr/particles_plot/image_raw")
        self.declare_parameter("frame_id", "particles_plot")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("manage_sensors_debug", False)

        topic = self.get_parameter("particles_topic").value
        image_topic = self.get_parameter("image_topic").value
        self._frame_id = self.get_parameter("frame_id").value
        publish_rate = float(self.get_parameter("publish_rate_hz").value)
        self._manage_sensors_debug = bool(
            self.get_parameter("manage_sensors_debug").value
        )

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = None
        self._fig, self._ax = plt.subplots(figsize=(5, 6), dpi=100)

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(FloatList, topic, self._cb, qos)
        self._image_pub = self.create_publisher(Image, image_topic, 1)
        self._set_param_client = self.create_client(SetParameters, SET_PARAM_SERVICE)

        period = 1.0 / max(publish_rate, 0.1)
        self.create_timer(period, self._publish_plot)
        self.get_logger().info(
            f"Subscribed to {topic}, publishing plot to {image_topic}"
        )

    def set_debug(self, value: bool) -> None:
        if not self._set_param_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                f"{SET_PARAM_SERVICE} not available. "
                "Particles only arrive if sensors debug is already on."
            )
            return

        req = SetParameters.Request()
        p = Parameter()
        p.name = "debug"
        p.value = ParameterValue()
        p.value.type = ParameterType.PARAMETER_BOOL
        p.value.bool_value = bool(value)
        req.parameters = [p]

        future = self._set_param_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"set debug={value} failed: {exc}")
            return
        if resp is None or not resp.results or not resp.results[0].successful:
            reason = resp.results[0].reason if resp and resp.results else "no response"
            self.get_logger().warn(f"set debug={value} rejected: {reason}")
        else:
            self.get_logger().info(f"Sensors debug -> {value}")

    def _cb(self, msg):
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.size < 5:
            return
        coords = arr[-2:]
        flat = arr[:-2]
        if flat.size % 3 != 0:
            return
        with self._lock:
            self._latest = (flat.reshape(3, -1), coords)

    def get_latest(self):
        with self._lock:
            if self._latest is None:
                return None
            p, c = self._latest
            return np.array(p, copy=True), np.array(c, copy=True)

    def _publish_plot(self):
        latest = self.get_latest()
        if latest is None:
            return
        particles, coords = latest
        _redraw(self._ax, particles, coords)
        self._fig.canvas.draw()
        rgba = np.asarray(self._fig.canvas.buffer_rgba())
        rgb = rgba[..., :3].copy()
        msg = self._bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        self._image_pub.publish(msg)


def _redraw(ax, particles, coords):
    ax.clear()
    ax.set_xlabel("Alignment")
    ax.set_ylabel("Distance along path [m]")
    ax.set_xlim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.scatter(
        particles[1], particles[0],
        c=particles[2], cmap="tab10", s=6, alpha=0.6,
        vmin=0, vmax=9,
    )
    ax.scatter(
        [coords[1]], [coords[0]],
        marker="*", s=220, c="red",
        edgecolors="black", linewidths=1.0, zorder=5,
    )
    d_est = float(coords[0])
    d_min = float(np.min(particles[0]))
    d_max = float(np.max(particles[0]))
    half_span = max(d_est - d_min, d_max - d_est) * 1.2
    ax.set_ylim(d_est - max(half_span, 0.5), d_est + max(half_span, 0.5))
    ax.set_title(
        f"N={particles.shape[1]}  estimate: d={coords[0]:.2f} m, a={coords[1]:+.3f}"
    )


def main():
    rclpy.init()
    node = ParticlesViewer()
    if node._manage_sensors_debug:
        node.set_debug(True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._manage_sensors_debug:
            node.set_debug(False)
        plt.close(node._fig)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
