#!/usr/bin/env python3
import numpy as np
from copy import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

from pfvtr.msg import FeaturesList, ImageList, Features, SensorsInput, Histogram
from pfvtr.srv import SetCameraTopic
from backends.siamese.siamese import SiameseCNN
from backends.siamese.siamfeature import SiamFeature
from camera_input import (
    camera_message_type,
    parse_camera_msg,
    resolve_camera_transport,
)

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


# Network hyperparameters (same as ROS1)
PAD = 32
NETWORK_DIVISION = 8
RESIZE_W = 512


class RepresentationMatching(Node):
    def __init__(self):
        super().__init__("representation_matching")
        self.get_logger().info("Sensor processing started!")

        self.declare_parameter("camera_topic", "/robot1/camera1/image")
        self.declare_parameter("camera_transport", "auto")
        self.declare_parameter("matching_type", "siam")
        self.declare_parameter("model_path", "")

        camera_topic = self.get_parameter("camera_topic").value
        self._camera_transport_param = self.get_parameter("camera_transport").value
        matching_type = self.get_parameter("matching_type").value
        model_path = self.get_parameter("model_path").value
        if len(model_path) == 0:
            model_path = None

        self.align_abs = None
        if matching_type == "siam_f":
            self.align_abs = SiamFeature(padding=PAD, resize_w=RESIZE_W, path_to_model=model_path)
        if matching_type == "siam":
            self.align_abs = SiameseCNN(padding=PAD, resize_w=RESIZE_W, path_to_model=model_path)
        if self.align_abs is None:
            raise Exception("Invalid matching scheme - edit launch file!")


        self.last_live = None
        self.sns_in_msg = None

        self.bridge = CvBridge()

        # Pre-warm the CNN before any subscription fires. The first GPU
        # inference pays a one-time cost (CUDA context, kernel JIT, cuDNN
        # algorithm selection, model-to-GPU transfer) that on a Jetson can
        # run 5-15s. Absorbing it here means the first real camera frame
        # hits an already-warm model, so the matching pipeline starts
        # flowing immediately once the repeater is active — rather than
        # the robot traversing several metres before alignment kicks in.
        try:
            dummy = np.zeros((RESIZE_W, RESIZE_W, 3), dtype=np.uint8)
            dummy_msg = self.bridge.cv2_to_imgmsg(dummy, encoding="rgb8")
            warm_in = ImageList()
            warm_in.data = [dummy_msg]
            t0 = self.get_clock().now()
            _ = self.align_abs._to_feature(warm_in)
            dt = (self.get_clock().now() - t0).nanoseconds / 1e9
            self.get_logger().warn(f"NN warmed up in {dt:.2f}s")
        except Exception as e:
            self.get_logger().warn(
                f"NN warmup failed (will warm on first frame): {e}"
            )

        # Two subscriptions, two *separate* mutually-exclusive callback groups.
        # A single shared group would serialise image_parserCB (slow CNN) with
        # map_parserCB (tiny assignment), starving map_parserCB whenever the
        # camera pushes frames faster than the CNN can process them, which
        # leaves self.sns_in_msg = None forever and silently disables matching.
        # Each callback still runs non-reentrant with itself, but the two can
        # now run in parallel on the MultiThreadedExecutor's threads.
        self.pub = self.create_publisher(FeaturesList, "live_representation", SYNC_FEEDER_QOS)
        self.pub_match = self.create_publisher(SensorsInput, "matched_repr", NAVIGATION_QOS)

        self._default_camera_topic = camera_topic
        self._camera_cb_group = get_exclusive_callback_group()
        self.sub = None
        self._active_camera_topic = ""
        self._rebind_camera(camera_topic)

        self.map_sub = self.create_subscription(
            SensorsInput,
            "map_representations",
            self.map_parserCB,
            NAVIGATION_QOS,
            callback_group=get_exclusive_callback_group()
        )

        self.create_service(
            SetCameraTopic,
            "set_camera_topic",
            self._on_set_camera_topic,
        )

    def _rebind_camera(self, topic: str) -> bool:
        topic = topic.strip() if topic else ""
        if not topic:
            topic = self._default_camera_topic
        transport = resolve_camera_transport(self._camera_transport_param, topic)
        msg_type = camera_message_type(transport)
        if (
            topic == self._active_camera_topic
            and self.sub is not None
            and getattr(self, "_active_camera_transport", None) == transport
        ):
            return True
        if self.sub is not None:
            self.destroy_subscription(self.sub)
            self.sub = None
        if transport == "compressed":
            cb = self.compressed_image_parserCB
        else:
            cb = self.image_parserCB
        self.sub = self.create_subscription(
            msg_type,
            topic,
            cb,
            NAVIGATION_QOS,
            callback_group=self._camera_cb_group,
        )
        self._active_camera_topic = topic
        self._active_camera_transport = transport
        self.get_logger().warn(
            f"Camera subscription bound to '{topic}' ({transport}, {msg_type.__name__})"
        )
        return True

    def _on_set_camera_topic(self, request: SetCameraTopic.Request, response: SetCameraTopic.Response):
        response.success = self._rebind_camera(request.topic)
        return response

    def compressed_image_parserCB(self, image: CompressedImage):
        self._process_camera_frame(image)

    def image_parserCB(self, image: Image):
        self._process_camera_frame(image)

    def _process_camera_frame(self, image):
        img_msg, _ = parse_camera_msg(image, self.bridge)
        if img_msg is None:
            self.get_logger().warn("camera frame decode failed — skipping")
            return
        msg = ImageList()
        msg.data = [img_msg]
        live_feature = self.align_abs._to_feature(msg)
        tmp_sns_in = copy(self.sns_in_msg)

        if self.last_live is None:
            self.last_live = live_feature[0]
        out = FeaturesList()
        out.header = img_msg.header
        out.data = [live_feature[0]]
        self.pub.publish(out)

        if tmp_sns_in is None:
            return

            # match live vs. live map, live vs last live, live vs maps
        ext_tensor = [*tmp_sns_in.map_features, self.last_live]
        align_in = SensorsInput()
        align_in.map_features = ext_tensor
        align_in.live_features = live_feature
        out = self.align_abs.process_msg(align_in)

        align_out = SensorsInput()

        live_hist = np.array(out[-1])  # all live map distances vs live img
        map_hist = np.array(out[:-1])

        align_out.header = img_msg.header
        align_out.live_histograms = [
            Histogram(values=list(live_hist.flatten()), shape=list(live_hist.shape))
        ]
        align_out.map_histograms = [
            Histogram(values=list(map_hist.flatten()), shape=list(map_hist.shape))
        ]
        align_out.map_distances = tmp_sns_in.map_distances
        align_out.map_transitions = tmp_sns_in.map_transitions
        align_out.map_timestamps = tmp_sns_in.map_timestamps
        align_out.map_num = tmp_sns_in.map_num
        align_out.map_similarity = tmp_sns_in.map_similarity  # TODO: this is not received from repeater yet!
        align_out.map_offset = tmp_sns_in.map_offset

        # rospy.logwarn("sending: " + str(hists.shape) + " " + str(tmp_sns_in.map_distances))
        self.pub_match.publish(align_out)
        self.last_live = live_feature[0]

    def map_parserCB(self, sns_in: SensorsInput):
        self.sns_in_msg = sns_in


def main():
    rclpy.init()
    node = RepresentationMatching()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
