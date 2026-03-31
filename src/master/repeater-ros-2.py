#!/usr/bin/env python3
import os
import time
import threading

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.serialization import deserialize_message

import rosbag2_py

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge

from pfvtr.action import MapRepeater
from pfvtr.msg import SensorsInput, SensorsOutput, Features, Histogram, DistancedTwist
from pfvtr.srv import SetDist, SetClockGain, StopRepeater



_bridge = CvBridge()

def parse_camera_msg(msg: Image) -> Image:
    img = _bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    if "bgr" in msg.encoding:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_msg = _bridge.cv2_to_imgmsg(img, encoding="rgb8")
    img_msg.header = msg.header
    return img_msg



def load_map(mappaths, images, distances, trans, times, source_align, logger):
    if "," in mappaths:
        mappaths = mappaths.split(",")
    else:
        mappaths = [mappaths]

    for map_idx, mappath in enumerate(mappaths):
        tmp = []
        for file in list(os.listdir(mappath)):
            if file.endswith(".npy"):
                tmp.append(file[:-4])
        logger.warn(str(len(tmp)) + " images found in the map")
        tmp.sort(key=lambda x: float(x))
        tmp_images = []
        tmp_distances = []
        tmp_trans = []
        tmp_times = []
        tmp_align = []
        source_map = None

        for idx, dist in enumerate(tmp):
            tmp_distances.append(float(dist))
            with open(os.path.join(mappath, dist + ".npy"), "rb") as fp:
                map_point = np.load(fp, allow_pickle=True, fix_imports=False).item(0)
                r = map_point["representation"]
                ts = map_point["timestamp"]
                diff_hist = map_point["diff_hist"]
                if map_point["source_map_align"] is None:
                    sm = mappath
                else:
                    sm = map_point["source_map_align"][0]

                if source_map is None:
                    source_map = sm
                if sm != source_map:
                    logger.warn("Multimap with invalid target!" + str(mappath))
                    raise Exception("Invalid map combination")

                if map_point["source_map_align"] is not None:
                    align = map_point["source_map_align"][1]
                else:
                    align = 0

                feature = Features()

                feature.shape = list(r[0].shape)
                feature.values = list(r[0].flatten())
                feature.descriptors = r[1]
                tmp_images.append(feature)
                tmp_times.append(ts)
                tmp_align.append(align)
                if diff_hist is not None:
                    tmp_trans.append(diff_hist)

                logger.info("Loaded feature: " + dist + ".npy")


        tmp_times[-1] = tmp_times[-2] + (tmp_times[-2] - tmp_times[-3])
        images.append(tmp_images)
        distances.append(tmp_distances)
        trans.append(tmp_trans)
        times.append(tmp_times)
        source_align.append(tmp_align)
        logger.warn("Whole map " + str(mappath) + " sucessfully loaded")


class RepeaterServer(Node):
    def __init__(self):
        super().__init__("repeater")

        self.cb_group = ReentrantCallbackGroup()

        self.img = None
        self.mapName = ""
        self.mapStep = None
        self.nextStep = 0
        self.isRepeating = False
        self.endPosition = 1.0
        self.clockGain = 1.0
        self.curr_dist = 0.0
        self.map_images = []
        self.map_distances = []
        self.map_alignments = []
        self.action_dists = None
        self.map_times = []
        self.actions = []
        self.map_publish_span = 1
        self.map_transitions = []
        self.use_distances = False
        self.distance_finish_offset = 0.2
        self.curr_map = 0
        self.map_num = 0
        self.last_map = 0
        self.nearest_map_img = -1
        self.null_cmd = False
        self.savedOdomTopic = ""

        self._active_goal_handle = None


        self.get_logger().debug("Waiting for services to become available...")

        self.distance_reset_cli = self.create_client(SetDist, "repeat/set_dist", callback_group=self.cb_group)
        self.align_reset_cli = self.create_client(SetDist, "repeat/set_align", callback_group=self.cb_group)

        while not self.distance_reset_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for repeat/set_dist...")
        while not self.align_reset_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for repeat/set_align...")


        self.create_service(SetClockGain, "set_clock_gain", self.setClockGain, callback_group=self.cb_group)
        self.create_service(StopRepeater, "stop_repeater", self.stopService, callback_group=self.cb_group)
        self.distance_sub = self.create_subscription(
            SensorsOutput, "repeat/output_dist", self.distanceCB, 1, callback_group=self.cb_group
        )

        self.sensors_pub = self.create_publisher(SensorsInput, "map_representations", 1)

        self.joy_topic = "map_vel"
        self.joy_pub = self.create_publisher(Twist, self.joy_topic, 1)


        self._action_server = ActionServer(
            self,
            MapRepeater,
            "repeater",
            execute_callback=self.actionCB,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
            callback_group=self.cb_group
        )

        self.get_logger().warn("Repeater started, awaiting goal")


    def setClockGain(self, req, resp):
        self.clockGain = req.gain
        return resp

    def stopService(self, req, resp):
        self.isRepeating = False
        self.get_logger().warn("Received stop request!")
        return resp


    def goal_cb(self, goal_request):
        if self.isRepeating:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle):
        return CancelResponse.ACCEPT

    def pubSensorsInput(self):
        if not self.isRepeating:
            return
        if len(self.map_images) > 0:
            #self.get_logger().warn(self.map_distances)
            # Load data from each map the map
            features = []
            distances = []
            timestamps = []
            offsets = []
            transitions = []
            last_nearest_img = self.nearest_map_img
            map_indices = []
            for map_idx in range(self.map_num):
                self.nearest_map_img = int(np.argmin(abs(self.curr_dist - np.array(self.map_distances[map_idx]))))
                # allow only move in map by one image per iteration
                lower_bound = max(0, self.nearest_map_img - self.map_publish_span)
                upper_bound = min(self.nearest_map_img + self.map_publish_span + 1,
                                  len(self.map_distances[map_idx]))

                features.extend(self.map_images[map_idx][lower_bound:upper_bound])
                distances.extend(self.map_distances[map_idx][lower_bound:upper_bound])
                timestamps.extend(self.map_times[map_idx][lower_bound:upper_bound])
                offsets.extend(self.map_alignments[map_idx][lower_bound:upper_bound])
                transitions.extend(self.map_transitions[map_idx][lower_bound:upper_bound - 1])
                map_indices.extend([map_idx for i in range(upper_bound - lower_bound)])

            if self.nearest_map_img != last_nearest_img:
                self.get_logger().info(
                    "matching image " + str(self.map_distances[-1][self.nearest_map_img]) +
                    " at distance " + str(self.curr_dist)
                )

            transitions = np.array(transitions)
            sns_in = SensorsInput()
            sns_in.header.stamp = self.get_clock().now().to_msg()
            sns_in.live_features = []
            sns_in.map_features = features
            sns_in.map_distances = distances

            sns_in.map_transitions = [
                Histogram(
                    values=list(transitions.flatten().astype(np.float32)),
                    shape=list(transitions.shape)
                )
            ]

            sns_in.map_timestamps = timestamps
            sns_in.map_num = self.map_num
            # TODO: sns_in.map_similarity
            sns_in.map_offset = offsets

            self.sensors_pub.publish(sns_in)
            self.last_map = self.curr_map


    def distanceCB(self, msg: SensorsOutput):
        if self.isRepeating is False:
            return

        # if self.img is None:
        #     rospy.logwarn("Warning: no image received")

        self.curr_dist = msg.output
        self.curr_map = msg.map

        if (self.curr_dist >= (
                self.map_distances[self.curr_map][-1] - self.distance_finish_offset) and self.use_distances) or \
                (self.endPosition != 0.0 and self.endPosition < self.curr_dist):
            self.get_logger().warn("GOAL REACHED, STOPPING REPEATER")
            self.isRepeating = False
            if self.use_distances:
                self.action_dists = []
                self.actions = []
            self.shutdown()

        if self.use_distances:
            self.play_closest_action()

        self.pubSensorsInput()


    def goalValid(self, goal) -> bool:
        if goal.map_name == "":
            self.get_logger().warn("Goal missing map name")
            return False
        # if not os.path.isdir(goal.mapName):
        #     rospy.logwarn("Can't find map directory")
        #     return False
        # if not os.path.isfile(os.path.join(goal.mapName, goal.mapName + ".bag")):
        #     rospy.logwarn("Can't find commands")
        #     return False
        # if not os.path.isfile(os.path.join(goal.mapName, "params")):
        #     rospy.logwarn("Can't find params")
        #     return False
        if goal.start_pos < 0:
            self.get_logger().warn("Invalid (negative) start position). Use zero to start at the beginning")
            return False
        if goal.start_pos > goal.end_pos:
            self.get_logger().warn("Start position greater than end position")
            return False
        return True


    def parseParams(self, filename: str):
        with open(filename, "r") as f:
            data = f.read()
        data = data.split("\n")
        data = filter(None, data)
        for line in data:
            line = line.split(" ")
            if "stepSize" in line[0]:
                self.get_logger().debug("Setting step size to: %s" % (line[1]))
                self.mapStep = float(line[1])
            if "odomTopic" in line[0]:
                self.get_logger().debug("Saved odometry topic is: %s" % (line[1]))
                self.savedOdomTopic = line[1]


    def checkShutdown(self):
        if self._active_goal_handle is not None and self._active_goal_handle.is_cancel_requested:
            self.shutdown()
            try:
                self._active_goal_handle.canceled()
            except Exception:
                pass
            self._active_goal_handle = None

    def shutdown(self):
        self.isRepeating = False

    def _open_bag2_reader(self, bag_uri: str) -> rosbag2_py.SequentialReader:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )
        reader.open(storage_options, converter_options)
        return reader


    def parse_rosbag(self, bag_uri: str):
        self.get_logger().warn("Starting to parse the actions")
        self.action_dists = []
        self.actions = []

        reader = self._open_bag2_reader(bag_uri)
        topics_info = reader.get_all_topics_and_types()
        type_by_name = {t.name: t.type for t in topics_info}

        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            if topic != "/recorded_actions":
                continue

            expected = "pfvtr/msg/DistancedTwist"
            if type_by_name.get(topic, "") != expected:
                raise RuntimeError(f"Unexpected type on {topic}: {type_by_name.get(topic,'')}")


            msg = deserialize_message(data, DistancedTwist)

            if self.null_cmd and msg.twist.linear.x == 0.0 and msg.twist.linear.y == 0.0 and msg.twist.angular.z == 0.0:
                continue

            self.action_dists.append(float(msg.distance))
            self.actions.append(msg.twist)

        self.action_dists = np.array(self.action_dists)
        self.get_logger().warn("Actions and distances successfully loaded!")

    def play_closest_action(self):
        if self.action_dists is not None and len(self.action_dists) > 0 and self.isRepeating:
            distance_to_pos = abs(self.curr_dist - self.action_dists)
            closest_idx = int(np.argmin(distance_to_pos))
            self.joy_pub.publish(self.actions[closest_idx])
        else:
            self.get_logger().warn("No action available - stopping")
            req = SetDist.Request()
            req.dist = 0.0
            req.mode = 1
            self.align_reset_cli.call(req)
            self.joy_pub.publish(Twist())


    def replay_timewise(self, bag_uri: str):
        self.get_logger().warn("Starting")
        previousMessageTime = None
        expectedMessageTime = None
        start = self.get_clock().now()

        reader = self._open_bag2_reader(bag_uri)
        topics_info = reader.get_all_topics_and_types()
        type_by_name = {t.name: t.type for t in topics_info}

        while reader.has_next():
            topic, data, t_ns = reader.read_next()

            now = self.get_clock().now()
            ts = rclpy.time.Time(nanoseconds=int(t_ns))

            if previousMessageTime is None:
                previousMessageTime = ts
                expectedMessageTime = now
            else:
                simulatedTimeToGo = ts - previousMessageTime
                corrected = rclpy.duration.Duration(nanoseconds=int(simulatedTimeToGo.nanoseconds * self.clockGain))
                error = now - expectedMessageTime
                sleepTime = corrected - error
                expectedMessageTime = now + sleepTime

                # rospy.sleep(sleepTime)
                if sleepTime.nanoseconds > 0:
                    time.sleep(sleepTime.nanoseconds / 1e9)

                previousMessageTime = ts

            if topic == "/recorded_actions":
                expected = "pfvtr/msg/DistancedTwist"
                if type_by_name.get(topic, "") != expected:
                    raise RuntimeError(f"Unexpected type on {topic}: {type_by_name.get(topic,'')}")
                msg = deserialize_message(data, DistancedTwist)
                self.joy_pub.publish(msg.twist)
            else:

                pass

            if self.isRepeating is False:
                self.get_logger().info("stopped!")
                break

            self.checkShutdown()

        self.isRepeating = False
        dur = self.get_clock().now() - start
        self.get_logger().warn("Rosbag runtime: %f" % (dur.nanoseconds / 1e9))


    def actionCB(self, goal_handle):
        self._active_goal_handle = goal_handle
        goal = goal_handle.request

        self.get_logger().info("New goal received")
        result = MapRepeater.Result()

        if self.goalValid(goal) is False:
            self.get_logger().warn("Ignoring invalid goal")
            result.success = False
            goal_handle.succeed()
            return result

        map_name = goal.map_name.split(",")[0]
        self.parseParams(os.path.join(map_name, "params"))

        self.map_publish_span = int(goal.image_pub)


        req = SetDist.Request()
        req.dist = 0.0
        req.mode = 1
        self.align_reset_cli.call(req)

        self.endPosition = goal.end_pos
        self.nextStep = 0
        self.null_cmd = goal.null_cmd


        self.map_images = []
        self.map_distances = []
        self.action_dists = None
        self.actions = []
        self.map_transitions = []
        self.last_closest_idx = 0
        self.map_alignments = []
        self.map_times = []

        map_loader = threading.Thread(
            target=load_map,
            args=(goal.map_name, self.map_images, self.map_distances,
                  self.map_transitions, self.map_times, self.map_alignments,
                  self.get_logger())
        )
        map_loader.start()
        map_loader.join()
        self.map_num = len(self.map_images)

        self.get_logger().warn("Starting repeat")
        self.mapName = goal.map_name
        self.use_distances = goal.use_dist


        bag_uri = os.path.join(map_name, "bag")


        req = SetDist.Request()
        req.dist = float(goal.start_pos)
        req.mode = int(self.map_num)
        self.distance_reset_cli.call(req)

        self.curr_dist = float(goal.start_pos)
        time.sleep(2)

        self.get_logger().info("Repeating started!")
        self.isRepeating = True

        if self.use_distances:
            self.parse_rosbag(bag_uri)
            self.play_closest_action()
        else:
            self.replay_timewise(bag_uri)

        while self.isRepeating:
            time.sleep(1)
            self.checkShutdown()

        result.success = True
        goal_handle.succeed()
        return result


def main():
    rclpy.init()
    node = RepeaterServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
