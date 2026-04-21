#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import Twist, TwistStamped
from pfvtr.msg import SensorsOutput
from pfvtr.srv import SetClockGain
import controller

NAVIGATION_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

def get_exclusive_callback_group():
    return MutuallyExclusiveCallbackGroup()


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller")

        self.c = controller.Controller()

        self.declare_parameter("cmd_vel_topic", "/bluetooth_teleop/cmd_vel")
        self.declare_parameter("use_uncertainty", True)
        self.declare_parameter("turn_gain", 0.05)
        self.declare_parameter("velocity_gain", 1.0)

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        cb_group = get_exclusive_callback_group()

        self.pub = self.create_publisher(TwistStamped, cmd_vel_topic, NAVIGATION_QOS)

        self.sub_vel = self.create_subscription(TwistStamped, "map_vel", self.callbackVel, NAVIGATION_QOS, callback_group=cb_group)
        self.sub_corr = self.create_subscription(SensorsOutput, "repeat/output_align", self.callbackCorr, NAVIGATION_QOS, callback_group=cb_group)

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

    def callbackVel(self, msg: TwistStamped):
        # controller.process() operates on a plain Twist; unwrap, process,
        # then restamp on the way out so subscribers (robot base, mapmaker)
        # that expect TwistStamped on /cmd_vel still receive it.
        driven = self.c.process(msg.twist)
        out = TwistStamped()
        out.header = msg.header
        out.twist = driven
        self.pub.publish(out)

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
