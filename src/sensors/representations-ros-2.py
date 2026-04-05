#!/usr/bin/env python3
import numpy as np
from copy import copy

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data, QoSProfile

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from pfvtr.msg import FeaturesList, ImageList, Features, SensorsInput, Histogram


from backends.siamese.siamese import SiameseCNN
from backends.siamese.siamfeature import SiamFeature


# Network hyperparameters (same as ROS1)
PAD = 32
NETWORK_DIVISION = 8
RESIZE_W = 512


class RepresentationMatching(Node):
    def __init__(self):
        super().__init__("sensor_processing")
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
        self.cb_group = ReentrantCallbackGroup()

        img_qos = qos_profile_sensor_data
        map_qos = QoSProfile(depth=1)

        self.pub = self.create_publisher(FeaturesList, "live_representation", 1)
        self.pub_match = self.create_publisher(SensorsInput, "matched_repr", 1)

        self.sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_parserCB,
            img_qos,
            callback_group=self.cb_group
        )

        self.map_sub = self.create_subscription(
            SensorsInput,
            "map_representations",
            self.map_parserCB,
            map_qos,
            callback_group=self.cb_group
        )


    def parse_camera_msg(self, msg: Image):

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if "bgr" in msg.encoding:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if "rgba" in msg.encoding:
            img = img[..., :3]
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
        img_msg.header = msg.header
        return img_msg, img

    def image_parserCB(self, image: Image):
        img_msg, _ = self.parse_camera_msg(image)
        msg = ImageList()
        msg.data = [img_msg]
        # msg = ImageList([img_msg])
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
            Histogram(live_hist.flatten(), live_hist.shape)]
        align_out.map_histograms = [Histogram(map_hist.flatten(), map_hist.shape)]
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
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
