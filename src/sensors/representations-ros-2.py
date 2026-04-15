#!/usr/bin/env python3
import numpy as np
from copy import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from pfvtr.msg import FeaturesList, ImageList, Features, SensorsInput, Histogram
from backends.siamese.siamese import SiameseCNN
from backends.siamese.siamfeature import SiamFeature

NAVIGATION_QOS = QoSProfile(
    depth=1,
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
        self.declare_parameter("matching_type", "siam")
        self.declare_parameter("model_path", "")

        camera_topic = self.get_parameter("camera_topic").value
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
        cb_group = get_exclusive_callback_group()

        self.pub = self.create_publisher(FeaturesList, "live_representation", NAVIGATION_QOS)
        self.pub_match = self.create_publisher(SensorsInput, "matched_repr", NAVIGATION_QOS)

        self.sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_parserCB,
            NAVIGATION_QOS,
            callback_group=cb_group
        )

        self.map_sub = self.create_subscription(
            SensorsInput,
            "map_representations",
            self.map_parserCB,
            NAVIGATION_QOS,
            callback_group=cb_group
        )

    def parse_camera_msg(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return None, None

        if img is None:
            self.get_logger().warn("Received empty image from cv_bridge")
            return None, None

        if hasattr(img, "size") and img.size == 0:
            self.get_logger().warn("Received zero-sized image")
            return None, None

        if "bgr" in msg.encoding:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif "rgba" in msg.encoding:
            if img.ndim == 3 and img.shape[2] >= 3:
                img = img[..., :3]
            else:
                self.get_logger().warn(f"Unexpected rgba image shape: {getattr(img, 'shape', None)}")
                return None, None
        elif "rgb" in msg.encoding:
            pass
        else:
            self.get_logger().warn(f"Unexpected image encoding: {msg.encoding}")

        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
        except Exception as e:
            self.get_logger().warn(f"cv2_to_imgmsg failed: {e}")
            return None, None

        img_msg.header = msg.header
        return img_msg, img

    def image_parserCB(self, image: Image):
        img_msg, _ = self.parse_camera_msg(image)
        msg = ImageList()
        msg.data = [img_msg]
        live_feature = self.align_abs._to_feature(msg)
        tmp_sns_in = copy(self.sns_in_msg)

        if self.last_live is None:
            self.last_live = live_feature[0]
        out = FeaturesList()
        out.header = image.header
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

        align_out.header = image.header
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
