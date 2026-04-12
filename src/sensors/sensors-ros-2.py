#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from pfvtr.srv import Alignment
from sensor_processing import BearnavClassic, PF2D, VisualOnly, NNPolicy
from rclpy.executors import MultiThreadedExecutor

from backends.odometry.odom_dist import OdometryAbsolute, OdometryRelative
from backends.siamese.siamese import SiameseCNN
from backends.siamese.siamfeature import SiamFeature
from backends.crosscorrelation.crosscorr import CrossCorrelation

# Network hyperparameters
PAD = 32
NETWORK_DIVISION = 8
RESIZE_W = 512


class SensorProcessingNode(Node):
    def __init__(self):
        super().__init__("sensor_processing")
        self.get_logger().info("Sensor processing started!")


        self.declare_parameter("odom_topic", "/robot1/odometry")
        self.declare_parameter("particle_num", 600)
        self.declare_parameter("odom_error", 0.025)
        self.declare_parameter("dist_init_std", 1.0)
        self.declare_parameter("align_beta", 10.0)
        self.declare_parameter("align_init_std", 1.0)
        self.declare_parameter("choice_beta", 2.5)
        self.declare_parameter("add_random", 0.01)
        self.declare_parameter("matching_type", "siam")
        self.declare_parameter("model_path", "")

        odom_topic = self.get_parameter("odom_topic").value
        particle_num = int(self.get_parameter("particle_num").value)
        odom_error = float(self.get_parameter("odom_error").value)
        dist_init_std = float(self.get_parameter("dist_init_std").value)
        align_beta = float(self.get_parameter("align_beta").value)
        align_init_std = float(self.get_parameter("align_init_std").value)
        choice_beta = float(self.get_parameter("choice_beta").value)
        add_random = float(self.get_parameter("add_random").value)
        matching_type = self.get_parameter("matching_type").value
        model_path = self.get_parameter("model_path").value
        if len(model_path) == 0:
            model_path = None


        align_abs = None
        if matching_type == "siam_f":
            align_abs = SiamFeature(padding=PAD, resize_w=RESIZE_W, path_to_model=model_path)
        if matching_type == "siam":
            align_abs = SiameseCNN(padding=PAD, resize_w=RESIZE_W, path_to_model=model_path)
        if align_abs is None:
            raise Exception("Invalid matching scheme - edit launch file!")

        align_rel = CrossCorrelation(padding=PAD, network_division=NETWORK_DIVISION, resize_w=RESIZE_W, logger=self.get_logger())
        dist_abs = OdometryAbsolute(logger=self.get_logger())
        dist_rel = OdometryRelative(logger=self.get_logger())


        # BearnavClassic is currently only supported
        self.teach_fusion = BearnavClassic(self, "teach", align_abs, dist_abs, align_abs, align_abs)
        self._start_subscribes(
            fusion=self.teach_fusion,
            abs_align_topic="",
            abs_dist_topic=odom_topic,
            rel_dist_topic="",
            prob_dist_topic="",
            rel_align_service_name="local_alignment"
        )

        # 1) Bearnav classic - this method also needs publish span 0 in the repeater !!!
        self.repeat_fusion = BearnavClassic(
            node=self,
            type_prefix="repeat",
            abs_align_est=align_abs,
            abs_dist_est=dist_abs,
            rel_align_est=align_abs,
            repr_creator=align_abs
        )
        self._start_subscribes(
            fusion=self.repeat_fusion,
            abs_align_topic="matched_repr",
            abs_dist_topic=odom_topic,
            rel_dist_topic="",
            prob_dist_topic="",
            rel_align_service_name="local_alignment"
        )
        
        # 2) Particle filter 2D - parameters are really important
        # self.repeat_fusion = PF2D(
        #     node=self,
        #     type_prefix="repeat",
        #     particles_num=particle_num,
        #     odom_error=odom_error,
        #     odom_init_std=dist_init_std,
        #     align_beta=align_beta,
        #     align_init_std=align_init_std,
        #     particles_frac=1,
        #     choice_beta=choice_beta,
        #     add_random=add_random,
        #     debug=True,
        #     abs_align_est=align_abs,
        #     rel_align_est=align_rel,
        #     rel_dist_est=dist_rel,
        #     repr_creator=align_abs
        # )
        # self._start_subscribes(
        #     fusion=self.repeat_fusion,
        #     abs_align_topic="matched_repr",
        #     abs_dist_topic="",
        #     rel_dist_topic=odom_topic,
        #     prob_dist_topic="",
        #     rel_align_service_name="local_alignment"
        # )

    def _start_subscribes(self, fusion,
                         abs_align_topic, abs_dist_topic, rel_dist_topic, prob_dist_topic,
                         rel_align_service_name):

        q1 = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        if fusion.abs_align_est is not None and len(abs_align_topic) > 0:
            self.create_subscription(
                fusion.abs_align_est.supported_message_type,
                abs_align_topic,
                fusion.process_abs_alignment,
                q1
            )

        if fusion.abs_dist_est is not None and len(abs_dist_topic) > 0:
            self.create_subscription(
                fusion.abs_dist_est.supported_message_type,
                abs_dist_topic,
                fusion.process_abs_distance,
                q1
            )

        if fusion.rel_dist_est is not None and len(rel_dist_topic) > 0:
            self.create_subscription(
                fusion.rel_dist_est.supported_message_type,
                rel_dist_topic,
                fusion.process_rel_distance,
                q1
            )

        if fusion.prob_dist_est is not None and len(prob_dist_topic) > 0:
            self.create_subscription(
                fusion.prob_dist_est.supported_message_type,
                prob_dist_topic,
                fusion.process_prob_distance,
                q1
            )

        if fusion.rel_align_est is not None and len(rel_align_service_name) > 0:
            self.create_service(
                Alignment,
                f"{fusion.type_prefix}/{rel_align_service_name}",
                fusion.process_rel_alignment
            )


def main():
    rclpy.init()
    node = SensorProcessingNode()
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        executor.add_node(node)
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
