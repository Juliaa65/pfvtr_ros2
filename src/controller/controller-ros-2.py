#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import Twist
from pfvtr.msg import SensorsOutput
from pfvtr.srv import SetClockGain

import controller


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller")

        self.c = controller.Controller()

        self.declare_parameter("cmd_vel_topic", "/bluetooth_teleop/cmd_vel")
        self.declare_parameter("use_uncertainty", True)
        self.declare_parameter("turn_gain", 2.0)
        self.declare_parameter("velocity_gain", 1.0)

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.sub_vel = self.create_subscription(Twist, "map_vel", self.callbackVel, 10)
        self.sub_corr = self.create_subscription(SensorsOutput, "repeat/output_align", self.callbackCorr, 10)

        self.gain_client = self.create_client(SetClockGain, "set_clock_gain")
        self._apply_controller_params()

        self.add_on_set_parameters_callback(self.callbackReconfigure)

    def _apply_controller_params(self):

        velocity_gain = float(self.get_parameter("velocity_gain").value)
        turn_gain = float(self.get_parameter("turn_gain").value)
        use_uncertainty = bool(self.get_parameter("use_uncertainty").value)

        if hasattr(self.c, "reconfig"):
            self.c.reconfig({
                "velocity_gain": velocity_gain,
                "turn_gain": turn_gain,
                "use_uncertainty": use_uncertainty
            })

        self._set_clock_gain_from_velocity(velocity_gain)

    def callbackVel(self, msg: Twist):
        driven = self.c.process(msg)
        self.pub.publish(driven)

    def callbackCorr(self, msg: SensorsOutput):
        self.c.correction(msg)

    def callbackReconfigure(self, params):
        for p in params:
            if p.name in ("velocity_gain", "turn_gain", "use_uncertainty"):
                continue
            if p.name == "cmd_vel_topic":
                return SetParametersResult(successful=False, reason="cmd_vel_topic cannot be changed at runtime")

        self._apply_controller_params()
        return SetParametersResult(successful=True)

    def _set_clock_gain_from_velocity(self, velocity_gain: float):
        try:
            gain = 1.0 / float(velocity_gain) if float(velocity_gain) != 0.0 else 1.0
        except Exception:
            self.get_logger().warn("Invalid velocity_gain, using gain=1.0")
            gain = 1.0

        if not self.gain_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn(
                "Service 'set_clock_gain' not available after 0.2s, skipping call"
            )
            return

        req = SetClockGain.Request()
        req.gain = float(gain)

        self.gain_client.call_async(req)
        self.get_logger().info(f"Called set_clock_gain with gain={gain}")


def main():
    rclpy.init()
    node = ControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
