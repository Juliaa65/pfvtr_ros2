#!/usr/bin/env python3
import os
import time
import shutil


import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.serialization import serialize_message
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

import rosbag2_py

from pfvtr.action import MapMaker
from pfvtr.msg import SensorsOutput, SensorsInput, DistancedTwist, Features, FeaturesList
from pfvtr.srv import SetDist, Alignment


TARGET_WIDTH = 512


def get_map_dists(mappath: str) -> np.ndarray:
    tmp = []
    for file in list(os.listdir(mappath)):
        if file.endswith(".npy"):
            tmp.append(file[:-4])
    tmp.sort(key=lambda x: float(x))
    if not tmp:
        raise Exception("Invalid source map (empty)")
    return np.array(tmp, dtype=float)


def numpy_to_feature(array):
    return Features(values=array[0].flatten().tolist(), shape=list(array[0].shape), descriptors=array[1])


def save_img(img_repr, image_msg: Image, header: Header, map_name: str,
             curr_dist, curr_hist, curr_align, source_map, save_img_flag: bool,
             bridge: CvBridge):
    filename = str(map_name) + "/" + str(curr_dist)
    ts = (header.stamp.sec, header.stamp.nanosec)


    struct_save = {
        "representation": img_repr,
        "timestamp": ts,
        "diff_hist": None,
        "source_map_align": None
    }
    if curr_hist is not None:
        struct_save["diff_hist"] = curr_hist
    if curr_align is not None and source_map is not None:
        struct_save["source_map_align"] = (source_map, curr_align)

    with open(filename + ".npy", "wb") as fp:
        np.save(fp, struct_save, fix_imports=False)

    ok = False
    if "rgb" in image_msg.encoding:
        cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        ok = cv2.imwrite(filename + ".jpg", cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))
    else:
        cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        ok = cv2.imwrite(filename + ".jpg", cv_img)

    if not ok:
        print(f"Failed to save image: {filename}.jpg")
    else:
        print(f"Saved image: {filename}.jpg")


class MapmakerServer(Node):
    def __init__(self):
        super().__init__("mapmaker")

        self.bridge = CvBridge()

        self.isMapping = False
        self.img_msg = None
        self.last_img_msg = None
        self.img_features = None
        self.last_img_features = None
        self.mapName = ""
        self.mapStep = 1.0
        self.nextStep = 0.0
        self.visual_turn = True
        self.max_trans = 0.3
        self.curr_trans = 0.0
        self.curr_hist = None
        self.last_saved_dist = None
        self.save_imgs = False
        self.header = None
        self.target_distances = None
        self.collected_distances = None
        self.dist = 0.0
        self.lastOdom = None
        self.curr_alignment = None
        self.source_map = None

        self.declare_parameter("cmd_vel_topic", "/bluetooth_teleop/cmd_vel")
        self.declare_parameter("odom_record_topic", "/odometry_publisher")
        self.declare_parameter("camera_topic", "/camera_front_publisher")

        self.joy_topic = self.get_parameter("cmd_vel_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.odom_record_topic = self.get_parameter("odom_record_topic").value
        self.camera_topic = self.get_parameter("camera_topic").value


        self.get_logger().info("Waiting for services to become available...")

        self.distance_reset_cli = self.create_client(SetDist, "teach/set_dist")
        self.align_reset_cli = self.create_client(SetDist, "teach/set_align")

        while not self.distance_reset_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().info("Waiting for teach/set_dist...")

        self.get_logger().info("teach/set_dist is available")

        while not self.align_reset_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().info("Waiting for teach/set_align...")

        self.get_logger().info("teach/set_align is available")

        self.call_setdist_blocking(self.distance_reset_cli, "teach/set_dist", 0.0, 1)
        self.call_setdist_blocking(self.align_reset_cli, "teach/set_align", 0.0, 1)

        self.local_align_cli = self.create_client(Alignment, "teach/local_alignment")
        if self.visual_turn:
            while not self.local_align_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for teach/local_alignment...")
            self.get_logger().warn("Local alignment service available for mapmaker")


        self.get_logger().debug("Subscribing to commands")
        self.joy_sub = self.create_subscription(Twist, self.joy_topic, self.joy_cb, 100) #buff size!!!!!!!

        if self.odom_record_topic:
            self.add_sub = self.create_subscription(Odometry, self.odom_record_topic, self.misc_cb, 10)

        self.get_logger().debug("Starting mapmaker action server")
        self._action_server = ActionServer(
            self,
            MapMaker,
            "mapmaker",
            execute_callback=self.action_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
        )

        self.get_logger().warn("Mapmaker starting subscribers")
        self._setup_teach_sync()


        self._bag_writer = None
        self._bag_open = False

        self.get_logger().warn("Mapmaker started, awaiting goal")

    def call_setdist_blocking(self, client, service_name: str, dist: float, map_num: int):
        req = SetDist.Request()
        req.dist = dist
        req.map_num = map_num

        self.get_logger().info(f"Calling {service_name}...")
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            self.get_logger().error(f"{service_name} did not complete in time")
            return None

        result = future.result()
        if result is None:
            self.get_logger().error(f"{service_name} failed")
            return None

        self.get_logger().info(f"Returned from {service_name}")
        return result

    def _setup_teach_sync(self):
        repr_sub = Subscriber(self, FeaturesList, "live_representation")
        cam_sub = Subscriber(self, Image, self.camera_topic)
        distance_sub = Subscriber(self, SensorsOutput, "teach/output_dist")

        self.synced_topics = ApproximateTimeSynchronizer(
            [repr_sub, distance_sub, cam_sub],
            queue_size=50,
            slop=2.5
        )
        self.synced_topics.registerCallback(self.distance_img_cb)

    def _setup_repeat_sync(self):
        repr_sub = Subscriber(self, FeaturesList, "live_representation")
        cam_sub = Subscriber(self, Image, self.camera_topic)
        distance_sub = Subscriber(self, SensorsOutput, "repeat/output_dist")
        align_sub = Subscriber(self, SensorsOutput, "repeat/output_align")

        self.synced_topics = ApproximateTimeSynchronizer(
            [repr_sub, distance_sub, align_sub, cam_sub],
            queue_size=10,
            slop=0.5
        )
        self.synced_topics.registerCallback(self.distance_wrapper_cb)


    def misc_cb(self, msg: Odometry):
        if self.isMapping:
            self.lastOdom = msg

    def distance_wrapper_cb(self, repr_msg: FeaturesList, dist_msg: SensorsOutput, align_msg: SensorsOutput, img: Image):
        self.curr_alignment = align_msg.output
        self.distance_img_cb(repr_msg, dist_msg, img)

    def distance_img_cb(self, repr_msg: FeaturesList, dist_msg: SensorsOutput, img: Image):
        if self.img_features is None:
            self.get_logger().warn("Mapmaker didn't receive the images successfully")

        feat0 = repr_msg.data[0]
        values = np.array(feat0.values).reshape(feat0.shape)
        self.img_features = [values, feat0.descriptors]

        self.img_msg = img
        self.header = repr_msg.header
        dist = float(dist_msg.output)
        self.dist = dist
        if not self.isMapping:
            return

        # obtain displacement between prev and new image
        if self.visual_turn and self.last_img_features is not None and dist:
            srv_msg = SensorsInput()
            srv_msg.map_features = [numpy_to_feature(self.last_img_features)]
            srv_msg.live_features = [numpy_to_feature(self.img_features)]
            try:
                req = Alignment.Request()
                req.input = srv_msg
                resp = self.local_align_cli.call(req)
                hist = resp.histograms[0].data
                half_size = np.size(hist) / 2.0
                self.curr_hist = hist
                self.curr_trans = -float(np.argmax(hist) - (np.size(hist) // 2.0)) / half_size
            except Exception as e:
                self.get_logger().warn(f"Service call failed: {e}")
        else:
            self.curr_trans = 0.0
            self.curr_hist = None

        # eventually save the image if conditions fulfilled
        if self.target_distances is not None and self.curr_hist is not None:
            desired_idx = int(np.argmin(abs(dist - np.array(self.target_distances))))
            self.last_img_features = self.img_features
            if self.collected_distances[desired_idx] == 0 and self.target_distances[desired_idx] <= dist:
                self.collected_distances[desired_idx] = 1
                save_img(
                    self.img_features, self.img_msg, self.header, self.mapName, dist,
                    self.curr_hist, self.curr_alignment, self.source_map, self.save_imgs, self.bridge
                )
                self.get_logger().info(f"Saved waypoint: {dist}, {self.curr_trans}")

        # save after fixed distance OR visual turn threshold
        if self.target_distances is None and (dist > self.nextStep or abs(self.curr_trans) > self.max_trans):
            self.nextStep = dist + self.mapStep
            self.last_img_features = self.img_features
            save_img(
                self.img_features, self.img_msg, self.header, self.mapName, dist,
                self.curr_hist, self.curr_alignment, self.source_map, self.save_imgs, self.bridge
            )
            self.get_logger().info(f"Saved waypoint: {dist}, {self.curr_trans}")

        if self.last_img_features is None:
            self.last_img_features = self.img_features

        self.checkShutdown()

    def joy_cb(self, msg: Twist):
        if self.isMapping:
            if self._bag_writer is None:
                return
            save_msg = DistancedTwist()
            save_msg.twist = msg
            save_msg.distance = float(self.dist)

            now_ns = self.get_clock().now().nanoseconds
            self._bag_writer.write("/recorded_actions", serialize_message(save_msg), now_ns)

            if self.lastOdom is not None:
                self._bag_writer.write("/recorded_odometry", serialize_message(self.lastOdom), now_ns)

    def goal_cb(self, goal_request):
        if goal_request.start:
            if self.isMapping:
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT
        else:
            if self.isMapping:
                return GoalResponse.ACCEPT
            return GoalResponse.REJECT

    def cancel_cb(self, goal_handle):
        return CancelResponse.ACCEPT

    def _bag_open_for_map(self, map_name: str):
        bag_dir = os.path.join(map_name, "bag")

        storage_options = rosbag2_py.StorageOptions(
            uri=bag_dir,
            storage_id="mcap"
        )

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )

        self._bag_writer = rosbag2_py.SequentialWriter()
        self._bag_writer.open(storage_options, converter_options)

        # Register topics
        self._bag_writer.create_topic(rosbag2_py.TopicMetadata(
            id=0,
            name="/recorded_actions",
            type="pfvtr/msg/DistancedTwist",
            serialization_format="cdr"
        ))
        self._bag_writer.create_topic(rosbag2_py.TopicMetadata(
            id=0,
            name="/recorded_odometry",
            type="nav_msgs/msg/Odometry",
            serialization_format="cdr"
        ))

        self._bag_open = True

    def _bag_close(self):
        # SequentialWriter closes on destruction
        self._bag_writer = None
        self._bag_open = False

    def action_cb(self, goal_handle):
        self._active_goal_handle = goal_handle
        goal = goal_handle.request
        result = MapMaker.Result()

        if goal.source_map != "":
            self.target_distances = []
            self.source_map = goal.source_map
            self.target_distances = get_map_dists(self.source_map)
            self.collected_distances = np.zeros_like(self.target_distances)
            self._setup_repeat_sync()
            self.get_logger().warn(f"mapmaker listening to distance callback of map {goal.source_map}")

        self.save_imgs = goal.save_imgs_for_viz

        if goal.map_name == "":
            self.get_logger().warn("Missing mapname, ignoring")
            result.success = False
            goal_handle.succeed()
            return result

        # start/stop mapping
        if goal.start == True:
            self.isMapping = False
            self.img_msg = None
            self.last_img_msg = None


            self.call_setdist_blocking(self.distance_reset_cli, "teach/set_dist", 0.0, 1)

            self.mapStep = goal.map_step
            if self.mapStep <= 0.0:
                self.get_logger().warn("Record step is not positive number - changing to 1.0m")
                self.mapStep = 1.0

            try:
                os.mkdir(goal.map_name)
                with open(os.path.join(goal.map_name, "params"), "w") as f:
                    f.write(f"stepSize: {self.mapStep}\n")
                    f.write(f"cmdVelTopic: {self.cmd_vel_topic}\n")
                    f.write(f"odomTopic: {self.odom_record_topic}\n")
            except Exception as e:
                self.get_logger().warn(f"Unable to create map directory, ignoring: {e}")
                result.success = False
                goal_handle.abort()
                return result

            self.get_logger().info("Starting mapping")
            self._bag_open_for_map(goal.map_name)

            self.mapName = goal.map_name
            self.nextStep = 0.0
            self.isMapping = True
            result.success = True
            goal_handle.succeed()
            return result

        else:
            # stop mapping
            if self.target_distances is None:
                if self.header is not None and self.img_msg is not None and self.img_features is not None:
                    save_img(
                        self.img_features, self.img_msg, self.header, self.mapName,
                        float(self.dist), self.curr_hist, self.curr_alignment,
                        self.source_map, self.save_imgs, self.bridge
                    )
                    self.get_logger().info(f"Creating final wp at dist: {self.dist}")
                else:
                    self.get_logger().warn(
                        "Skipping final waypoint save: no synchronized image/features/header received yet")

            self.get_logger().warn("Stopping Mapping")
            self.get_logger().info(f"Map saved under: '{os.path.abspath(self.mapName)}'")

            time.sleep(2)
            self.isMapping = False
            result.success = True
            goal_handle.succeed()

            self._bag_close()

            if self.target_distances is not None:
                self.get_logger().warn("Removing and copying action commands")
                dst_bag_dir = os.path.join(goal.map_name, "bag")
                src_bag_dir = os.path.join(goal.source_map, "bag")
                try:
                    if os.path.isdir(dst_bag_dir):
                        shutil.rmtree(dst_bag_dir)
                    shutil.copytree(src_bag_dir, dst_bag_dir)
                except Exception as e:
                    self.get_logger().warn(f"Failed to copy bag2 from source map: {e}")

            return result

    def checkShutdown(self):
        if self._active_goal_handle is not None and self._active_goal_handle.is_cancel_requested:
            self.shutdown()
            try:
                self._active_goal_handle.canceled()
            except Exception:
                pass
            self._active_goal_handle = None

    def shutdown(self):
        self.isMapping = False
        if self._bag_writer is not None:
            self._bag_close()


def main():
    rclpy.init()
    node = MapmakerServer()
    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
