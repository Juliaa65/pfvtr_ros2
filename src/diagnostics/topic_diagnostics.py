#!/usr/bin/env python3

from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosidl_runtime_py.utilities import get_message

from std_msgs.msg import String, Bool
from action_msgs.msg import GoalStatusArray


class SanityMonitorNode(Node):
    def __init__(self):
        super().__init__('sanity_monitor_node')

        self.declare_parameter('camera_topic', '/camera_front_publisher')
        self.declare_parameter('odom_topic', '/odometry_publisher')
        self.declare_parameter('live_repr_topic', '/pfvtr/live_representation')

        self.declare_parameter('cmd_vel_sub_topic', '/cmd_vel_subscriber')
        self.declare_parameter('map_repr_topic', '/pfvtr/map_representations')
        self.declare_parameter('map_vel_topic', '/pfvtr/map_vel')
        self.declare_parameter('matched_repr_topic', '/pfvtr/matched_repr')
        self.declare_parameter(
            'repeat_distance_topic',
            '/pfvtr/repeat/distance_remaining'
        )

        camera_topic = self.get_parameter('camera_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        live_repr_topic = self.get_parameter('live_repr_topic').value

        cmd_vel_sub_topic = self.get_parameter('cmd_vel_sub_topic').value
        map_repr_topic = self.get_parameter('map_repr_topic').value
        map_vel_topic = self.get_parameter('map_vel_topic').value
        matched_repr_topic = self.get_parameter('matched_repr_topic').value
        repeat_distance_topic = self.get_parameter('repeat_distance_topic').value

        self.timeout_sec = 3.0
        self.window_sec = 5.0

        self.runtime_states = {
            'teach_active': False,
            'repeat_active': False,
        }

        self.required_topics = {
            'camera': camera_topic,
            'odometry': odom_topic,
            'live_repr': live_repr_topic,
        }

        self.optional_topics = {
            'cmd_vel_sub': cmd_vel_sub_topic,
            'map_repr': map_repr_topic,
            'map_vel': map_vel_topic,
            'matched_repr': matched_repr_topic,
            'repeat_distance': repeat_distance_topic,
        }

        self.all_topics = {}
        self.all_topics.update(self.required_topics)
        self.all_topics.update(self.optional_topics)

        self.msg_times = {name: deque() for name in self.all_topics}
        self.last_msg_time = {name: None for name in self.all_topics}
        self.subscriptions_created = set()

        self.status_pub = self.create_publisher(String, '/sanity_check', 10)
        self.ready_pub = self.create_publisher(Bool, '/sanity_check/ready', 10)

        self.create_subscription(
            GoalStatusArray,
            '/pfvtr/mapmaker/_action/status',
            self.mapmaker_status_callback,
            10
        )

        self.create_subscription(
            GoalStatusArray,
            '/pfvtr/repeater/_action/status',
            self.repeater_status_callback,
            10
        )

        self.discovery_timer = self.create_timer(
            1.0,
            self.discover_and_subscribe
        )

        self.status_timer = self.create_timer(
            1.0,
            self.publish_status
        )

        self.get_logger().info('PFVTR sanity monitor started.')

    def mapmaker_status_callback(self, msg):
        self.runtime_states['teach_active'] = self.has_active_goal(msg)

    def repeater_status_callback(self, msg):
        self.runtime_states['repeat_active'] = self.has_active_goal(msg)

    def has_active_goal(self, msg):
        for status in msg.status_list:
            if status.status in [1, 2]:
                return True
        return False

    def discover_and_subscribe(self):
        topic_types = dict(self.get_topic_names_and_types())

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        for name, topic in self.all_topics.items():
            if name in self.subscriptions_created:
                continue

            if not topic:
                continue

            if topic not in topic_types:
                continue

            msg_type_name = topic_types[topic][0]

            try:
                msg_type = get_message(msg_type_name)

                self.create_subscription(
                    msg_type,
                    topic,
                    lambda msg, n=name: self.topic_callback(n),
                    qos
                )

                self.subscriptions_created.add(name)

                self.get_logger().info(
                    f'Subscribed to {topic} [{msg_type_name}]'
                )

            except Exception as e:
                self.get_logger().warn(
                    f'Could not subscribe to {topic}: {e}'
                )

    def topic_callback(self, topic_name):
        now = self.get_clock().now().nanoseconds / 1e9

        self.last_msg_time[topic_name] = now
        self.msg_times[topic_name].append(now)

        while (
            self.msg_times[topic_name]
            and now - self.msg_times[topic_name][0] > self.window_sec
        ):
            self.msg_times[topic_name].popleft()

    def calculate_frequency(self, topic_name):
        times = self.msg_times[topic_name]

        if len(times) < 2:
            return 0.0

        duration = times[-1] - times[0]

        if duration <= 0.0:
            return 0.0

        return (len(times) - 1) / duration

    def topic_status(self, name, now):
        topic = self.all_topics[name]

        if not topic:
            return f'{name}=DISABLED', True

        if name not in self.subscriptions_created:
            return f'{name}=NOT_FOUND({topic})', False

        last_time = self.last_msg_time[name]

        if last_time is None:
            return f'{name}=NO_MSG', False

        age = now - last_time
        freq = self.calculate_frequency(name)

        if age > self.timeout_sec:
            return f'{name}=TIMEOUT({age:.1f}s)', False

        return f'{name}={freq:.1f}Hz', True

    def publish_status(self):
        now = self.get_clock().now().nanoseconds / 1e9

        required_ok = True
        required_parts = []
        optional_parts = []

        for name in self.required_topics:
            text, ok = self.topic_status(name, now)
            required_parts.append(text)

            if not ok:
                required_ok = False

        for name in self.optional_topics:
            text, _ = self.topic_status(name, now)
            optional_parts.append(text)

        state = 'READY' if required_ok else 'NOT_READY'
        teach_state = 'ACTIVE' if self.runtime_states['teach_active'] else 'INACTIVE'
        repeat_state = 'ACTIVE' if self.runtime_states['repeat_active'] else 'INACTIVE'

        status_msg = String()
        status_msg.data = (
            f'{state}\n'
            f'teach={teach_state}\n'
            f'repeat={repeat_state}\n'
            f'required:\n  ' + '\n  '.join(required_parts) + '\n'
            f'optional:\n  ' + '\n  '.join(optional_parts)
        )

        ready_msg = Bool()
        ready_msg.data = required_ok

        self.status_pub.publish(status_msg)
        self.ready_pub.publish(ready_msg)


def main(args=None):
    rclpy.init(args=args)

    node = SanityMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()