#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult

from pfvtr.srv import Alignment
from sensor_processing import BearnavClassic, PF2D, VisualOnly  # , NNPolicy

NAVIGATION_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

SYNC_FEEDER_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

def get_exclusive_callback_group():
    return MutuallyExclusiveCallbackGroup()

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
        # PF2D output estimator. "kde" picks the dominant mode (correct under
        # multimodal posteriors). "weighted_mean" is the legacy centroid (drifts
        # between modes but well-tested in production).
        self.declare_parameter("position_estimator", "kde")
        # KDE-peak estimator hyperparameters (only consulted when
        # position_estimator=="kde", but declared unconditionally so launch
        # files don't need to know which estimator is active).
        self.declare_parameter("kde_grid_res", 64)
        self.declare_parameter("kde_align_span", 0.5)
        self.declare_parameter("kde_min_align_frac", 0.08)
        self.declare_parameter("matching_type", "siam")
        self.declare_parameter("model_path", "")
        # Repeat-phase fusion class. Two options:
        #   "classic" -> BearnavClassic, image-based correction (requires
        #                repeater image_pub == 0).
        #   "pf2d"    -> PF2D particle filter; uses particle_num, odom_error,
        #                dist_init_std, align_beta, align_init_std, choice_beta,
        #                add_random declared above (requires image_pub >= 1).
        # This is the *only* sensors-node parameter that is reconfigurable at
        # runtime: `ros2 param set /pfvtr/sensors navigation_method <classic|pf2d>`
        # (or the GUI's switch) tears down `self.repeat_fusion` and rebuilds
        # it in the new mode without restarting the node. After the swap the
        # user must re-issue a repeat goal so `image_pub` matches the new
        # mode and PF2D's particle cloud is re-seeded via set_distance.
        self.declare_parameter("navigation_method", "classic")

        # Hoisted onto self.* so _build_repeat_fusion can rebuild the fusion
        # at runtime without re-reading parameters or re-loading the Siamese
        # model on every swap.
        self.odom_topic = self.get_parameter("odom_topic").value
        self.particle_num = int(self.get_parameter("particle_num").value)
        self.odom_error = float(self.get_parameter("odom_error").value)
        self.dist_init_std = float(self.get_parameter("dist_init_std").value)
        self.align_beta = float(self.get_parameter("align_beta").value)
        self.align_init_std = float(self.get_parameter("align_init_std").value)
        self.choice_beta = float(self.get_parameter("choice_beta").value)
        self.add_random = float(self.get_parameter("add_random").value)
        self.position_estimator = self.get_parameter("position_estimator").value
        if self.position_estimator not in PF2D.POSITION_ESTIMATORS:
            raise Exception(
                f"Invalid position_estimator '{self.position_estimator}' "
                f"- must be one of {PF2D.POSITION_ESTIMATORS}"
            )
        self.kde_grid_res = int(self.get_parameter("kde_grid_res").value)
        self.kde_align_span = float(self.get_parameter("kde_align_span").value)
        self.kde_min_align_frac = float(self.get_parameter("kde_min_align_frac").value)
        matching_type = self.get_parameter("matching_type").value
        model_path = self.get_parameter("model_path").value
        if len(model_path) == 0:
            model_path = None
        self.navigation_method = self.get_parameter("navigation_method").value
        if self.navigation_method not in ("classic", "pf2d"):
            raise Exception(
                f"Invalid navigation_method '{self.navigation_method}' "
                "- must be 'classic' or 'pf2d'"
            )
        self.get_logger().info(f"Repeat-phase fusion: navigation_method={self.navigation_method}")


        self.align_abs = None
        if matching_type == "siam_f":
            self.align_abs = SiamFeature(padding=PAD, resize_w=RESIZE_W, path_to_model=model_path)
        if matching_type == "siam":
            self.align_abs = SiameseCNN(padding=PAD, resize_w=RESIZE_W, path_to_model=model_path)
        if self.align_abs is None:
            raise Exception("Invalid matching scheme - edit launch file!")

        self.align_rel = CrossCorrelation(padding=PAD, network_division=NETWORK_DIVISION, resize_w=RESIZE_W, logger=self.get_logger())
        # Separate OdometryAbsolute per fusion: both teach and repeat subscribe
        # to the odometry topic, and `process_abs_distance` only short-circuits
        # while `self.distance is None`. After a teach run, teach_fusion.distance
        # is set, so during repeat both callbacks fire and a shared accumulator
        # would be incremented twice per message.
        teach_dist_abs = OdometryAbsolute(logger=self.get_logger())
        self.repeat_dist_abs = OdometryAbsolute(logger=self.get_logger())
        self.dist_rel = OdometryRelative(logger=self.get_logger())


        # Teach fusion is fixed to BearnavClassic; its handles persist for
        # the node's lifetime and are never torn down by the swap path.
        self.teach_fusion = BearnavClassic(self, "teach", self.align_abs, teach_dist_abs, self.align_abs, self.align_abs)
        self._start_subscribes(
            fusion=self.teach_fusion,
            abs_align_topic="",
            abs_dist_topic=self.odom_topic,
            rel_dist_topic="",
            prob_dist_topic="",
            rel_align_service_name="local_alignment"
        )

        # Repeat fusion is built via a helper so it can be torn down and
        # rebuilt at runtime when navigation_method changes. The handles
        # are tracked separately so _do_swap_navigation_method can destroy
        # exactly the right ones without touching teach.
        self.repeat_fusion = self._build_repeat_fusion(self.navigation_method)
        self._repeat_subs, self._repeat_srvs = self._start_repeat_subscribes(
            self.repeat_fusion, self.navigation_method
        )

        # Param callback: validates new navigation_method values synchronously
        # and schedules the actual fusion-rebuild on a one-shot timer so it
        # runs after the parameter store commits. All other sensors-node
        # parameters are launch-only (rejected by the callback).
        self._pending_navigation_method = None
        self._swap_timer = None
        self.add_on_set_parameters_callback(self._on_set_parameters)

    def _build_repeat_fusion(self, method):
        """Construct repeat_fusion fresh for `method` ('classic' or 'pf2d').

        Reuses the shared estimators (align_abs, align_rel, repeat_dist_abs,
        dist_rel) created in __init__ so we don't reload the Siamese model on
        every swap. Caller is responsible for tearing down the previous
        fusion's ROS handles before calling this.
        """
        if method == "pf2d":
            return PF2D(
                node=self,
                type_prefix="repeat",
                particles_num=self.particle_num,
                odom_error=self.odom_error,
                odom_init_std=self.dist_init_std,
                align_beta=self.align_beta,
                align_init_std=self.align_init_std,
                particles_frac=1,
                choice_beta=self.choice_beta,
                add_random=self.add_random,
                debug=True,
                position_estimator=self.position_estimator,
                kde_grid_res=self.kde_grid_res,
                kde_align_span=self.kde_align_span,
                kde_min_align_frac=self.kde_min_align_frac,
                abs_align_est=self.align_abs,
                rel_align_est=self.align_rel,
                rel_dist_est=self.dist_rel,
                repr_creator=self.align_abs
            )
        elif method == "classic":
            return BearnavClassic(
                node=self,
                type_prefix="repeat",
                abs_align_est=self.align_abs,
                abs_dist_est=self.repeat_dist_abs,
                rel_align_est=self.align_abs,
                repr_creator=self.align_abs
            )
        else:
            raise Exception(f"Invalid navigation_method '{method}'")

    def _start_repeat_subscribes(self, fusion, method):
        # The two modes need different topic topology: pf2d takes relative
        # odometry (it integrates motion via process_rel_distance), classic
        # takes absolute distance (it short-circuits inside process_abs_distance
        # once self.distance is seeded).
        if method == "pf2d":
            return self._start_subscribes(
                fusion=fusion,
                abs_align_topic="matched_repr",
                abs_dist_topic="",
                rel_dist_topic=self.odom_topic,
                prob_dist_topic="",
                rel_align_service_name="local_alignment"
            )
        else:  # classic
            return self._start_subscribes(
                fusion=fusion,
                abs_align_topic="matched_repr",
                abs_dist_topic=self.odom_topic,
                rel_dist_topic="",
                prob_dist_topic="",
                rel_align_service_name="local_alignment"
            )

    def _start_subscribes(self, fusion,
                         abs_align_topic, abs_dist_topic, rel_dist_topic, prob_dist_topic,
                         rel_align_service_name):

        cb_group = get_exclusive_callback_group()
        subs = []
        srvs = []

        if fusion.abs_align_est is not None and len(abs_align_topic) > 0:
            subs.append(self.create_subscription(
                fusion.abs_align_est.supported_message_type,
                abs_align_topic,
                fusion.process_abs_alignment,
                SYNC_FEEDER_QOS,
                callback_group=cb_group
            ))

        if fusion.abs_dist_est is not None and len(abs_dist_topic) > 0:
            subs.append(self.create_subscription(
                fusion.abs_dist_est.supported_message_type,
                abs_dist_topic,
                fusion.process_abs_distance,
                SYNC_FEEDER_QOS,
                callback_group=cb_group
            ))

        if fusion.rel_dist_est is not None and len(rel_dist_topic) > 0:
            subs.append(self.create_subscription(
                fusion.rel_dist_est.supported_message_type,
                rel_dist_topic,
                fusion.process_rel_distance,
                SYNC_FEEDER_QOS,
                callback_group=cb_group
            ))

        if fusion.prob_dist_est is not None and len(prob_dist_topic) > 0:
            subs.append(self.create_subscription(
                fusion.prob_dist_est.supported_message_type,
                prob_dist_topic,
                fusion.process_prob_distance,
                SYNC_FEEDER_QOS,
                callback_group=cb_group
            ))

        if fusion.rel_align_est is not None and len(rel_align_service_name) > 0:
            srvs.append(self.create_service(
                Alignment,
                f"{fusion.type_prefix}/{rel_align_service_name}",
                fusion.process_rel_alignment
            ))

        return subs, srvs

    def _on_set_parameters(self, params):
        # Pre-set callback: only `navigation_method` is runtime-reconfigurable.
        # Reject any other param. The actual teardown/rebuild is deferred to
        # a one-shot timer so it runs *after* the parameter store commits and
        # outside this synchronous callback.
        for p in params:
            if p.name != "navigation_method":
                return SetParametersResult(
                    successful=False,
                    reason=f"{p.name} cannot be changed at runtime (sensors node)"
                )
            new_value = p.value
            if new_value not in ("classic", "pf2d"):
                return SetParametersResult(
                    successful=False,
                    reason=f"navigation_method must be 'classic' or 'pf2d', got '{new_value}'"
                )
            if new_value == self.navigation_method:
                continue  # no-op
            self._pending_navigation_method = new_value
            if self._swap_timer is None:
                self._swap_timer = self.create_timer(0.0, self._do_swap_navigation_method)
        return SetParametersResult(successful=True)

    def _do_swap_navigation_method(self):
        # rclpy has no native one-shot timer; cancel + destroy on first fire.
        if self._swap_timer is not None:
            self._swap_timer.cancel()
            self.destroy_timer(self._swap_timer)
            self._swap_timer = None
        if self._pending_navigation_method is None:
            return
        new_method = self._pending_navigation_method
        self._pending_navigation_method = None
        old_fusion = self.repeat_fusion

        # 1. Stop new callbacks from being scheduled by destroying the old
        #    subscriptions/services first. Already-running callbacks may still
        #    be in flight — the lock acquisition below waits for them.
        for s in self._repeat_subs:
            self.destroy_subscription(s)
        for s in self._repeat_srvs:
            self.destroy_service(s)
        self._repeat_subs = []
        self._repeat_srvs = []

        # 2. Wait for any in-flight fusion callback to release the lock, then
        #    tear down the fusion's own ROS handles created inside
        #    SensorFusion.__init__ (and PF2D's optional particles_pub).
        with old_fusion._particle_lock:
            self.destroy_publisher(old_fusion.output_dist)
            self.destroy_publisher(old_fusion.output_align)
            self.destroy_service(old_fusion.set_distance_srv)
            self.destroy_service(old_fusion.set_alignment_srv)
            if hasattr(old_fusion, "particles_pub"):
                self.destroy_publisher(old_fusion.particles_pub)

        # 3. Build the new fusion and re-subscribe.
        self.navigation_method = new_method
        self.repeat_fusion = self._build_repeat_fusion(new_method)
        self._repeat_subs, self._repeat_srvs = self._start_repeat_subscribes(
            self.repeat_fusion, new_method
        )

        self.get_logger().warn(
            f"navigation_method swapped: {old_fusion.__class__.__name__} -> "
            f"{self.repeat_fusion.__class__.__name__}. Re-issue your repeat "
            "goal so image_pub matches the new mode (classic: 0, pf2d: >=1) "
            "and PF2D's particle cloud is re-seeded via set_distance."
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
