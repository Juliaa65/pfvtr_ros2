#!/usr/bin/env python3
import os
import sys
import time
import math
import shutil
import threading
from queue import Queue

# camera_input is installed next to this script (lib/pfvtr) and lives in src/sensors.
for _pfvtr_py_dir in (
    os.path.dirname(os.path.abspath(__file__)),
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "sensors")
    ),
):
    if os.path.isdir(_pfvtr_py_dir) and _pfvtr_py_dir not in sys.path:
        sys.path.insert(0, _pfvtr_py_dir)


import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.serialization import serialize_message, deserialize_message
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from message_filters import ApproximateTimeSynchronizer, Cache, Subscriber
from cv_bridge import CvBridge

import rosbag2_py

from pfvtr.action import MapMaker
from pfvtr.msg import SensorsOutput, SensorsInput, DistancedTwist, Features, FeaturesList
from pfvtr.srv import SetDist, Alignment, SetCameraTopic
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

SYNC_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# Shallow queue for camera frames over Zenoh — avoids stacking stale JPEGs.
CAMERA_SYNC_QOS = QoSProfile(
    depth=2,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)


TARGET_WIDTH = 512


from maps_paths import map_path as _map_path, maps_dir


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


def _lookup_nearest(cache: Cache, stamp: Time, slop: Duration):
    """Return the cached message whose header stamp is nearest to ``stamp`` within ±slop."""
    t0 = stamp - slop
    t1 = stamp + slop
    best_msg = None
    best_dt_ns = None
    best_is_before = False
    for msg, msg_time in zip(cache.cache_msgs, cache.cache_times):
        if msg_time < t0 or msg_time > t1:
            continue
        dt_ns = abs((msg_time - stamp).nanoseconds)
        is_before = msg_time <= stamp
        if (
            best_msg is None
            or dt_ns < best_dt_ns
            or (dt_ns == best_dt_ns and is_before and not best_is_before)
        ):
            best_msg = msg
            best_dt_ns = dt_ns
            best_is_before = is_before
    return best_msg


def _lookup_miss_info(cache: Cache, stamp: Time, slop: Duration) -> dict:
    """Diagnostics when _lookup_nearest returns None."""
    cache_size = len(cache.cache_msgs)
    best_dt_ns = None
    for msg_time in cache.cache_times:
        dt_ns = abs((msg_time - stamp).nanoseconds)
        if best_dt_ns is None or dt_ns < best_dt_ns:
            best_dt_ns = dt_ns
    slop_sec = float(slop.nanoseconds) / 1e9
    if cache_size == 0:
        return {
            "cache_size": 0,
            "best_dt_sec": None,
            "slop_sec": slop_sec,
            "reason": "topic silent (cache empty)",
        }
    best_dt_sec = float(best_dt_ns) / 1e9
    if best_dt_sec > slop_sec:
        return {
            "cache_size": cache_size,
            "best_dt_sec": best_dt_sec,
            "slop_sec": slop_sec,
            "reason": f"stamp skew {best_dt_sec:.3f}s > slop {slop_sec:.3f}s",
        }
    return {
        "cache_size": cache_size,
        "best_dt_sec": best_dt_sec,
        "slop_sec": slop_sec,
        "reason": "no match in cache (unknown)",
    }


def save_img(img_repr, image_msg: Image, header: Header, map_name: str,
             curr_dist, curr_hist, curr_align, source_map, save_img_flag: bool,
             bridge: CvBridge):
    filename = _map_path(map_name, str(curr_dist))
    ts = header.stamp.sec + header.stamp.nanosec / 1e9


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

        # Background worker that serializes waypoint writes to disk so that
        # cv2.imwrite + np.save do not block the rclpy executor thread. Both
        # release the GIL during file I/O, so a plain daemon thread keeps
        # teach callbacks firing while the previous waypoint is still on disk.
        self._save_queue: Queue = Queue()
        self._save_worker = threading.Thread(
            target=self._save_worker_loop, daemon=True, name="mapmaker-save-worker"
        )
        self._save_worker.start()

        self.isMapping = False
        self.img_msg = None
        self.last_img_msg = None
        self.img_features = None
        self.last_img_features = None
        self.mapName = ""
        self.mapStep = 1.0
        self.nextStep = 0.0
        self._startup_skip_count = 0  # flush stale repr frames after START
        self._teach_bag_armed = False
        self._last_teach_lookup_warn_ns = 0
        self.visual_turn = True
        self.max_trans = 0.3
        self.curr_trans = 0.0
        self.curr_hist = None
        self.last_saved_dist = None
        self.last_action_dist = 0.0
        self.action_dist_step = 0.0001
        self.save_imgs = False
        self.header = None
        self.target_distances = None
        self.collected_distances = None
        self.dist = 0.0
        self.lastOdom = None
        self.lastGpsOdom = None
        self.curr_alignment = None
        self.source_map = None
        self.align_future = None
        self.align_in_progress = False

        self.declare_parameter("cmd_vel_topic", "/bluetooth_teleop/cmd_vel")
        self.declare_parameter("odom_record_topic", "/odometry_publisher")
        self.declare_parameter("gps_record_topic", "")
        self.declare_parameter("camera_topic", "/camera_front_publisher")
        self.declare_parameter("camera_transport", "raw")
        # Empty string means "no rear camera configured" — backward mapping
        # requests will be rejected in that case.
        self.declare_parameter("camera_back_topic", "")
        # Teach lookup buffers: repr triggers; dist/cam are looked up by stamp.
        # teach_cam_cache_size is in messages — raise if CNN latency * camera_hz
        # exceeds this (e.g. 60–90 for ~2s lag at 30 Hz).
        self.declare_parameter("teach_dist_cache_size", 1000)
        self.declare_parameter("teach_cam_cache_size", 10)
        self.declare_parameter("teach_lookup_slop_sec", 1.5)
        self.declare_parameter("record_debug", True)
        self.declare_parameter("record_debug_period_sec", 5.0)
        self.declare_parameter("teach_repr_trace", True)
        self.declare_parameter("teach_repr_log_every", 1)
        self.declare_parameter("teach_feed_watchdog_sec", 3.0)

        self.joy_topic = self.get_parameter("cmd_vel_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.odom_record_topic = self.get_parameter("odom_record_topic").value
        self.gps_record_topic = (
            self.get_parameter("gps_record_topic").value or ""
        ).strip()
        self.camera_topic = self.get_parameter("camera_topic").value
        self._camera_transport_param = self.get_parameter("camera_transport").value
        self.camera_back_topic = (
            self.get_parameter("camera_back_topic").value or ""
        ).strip()
        self._default_camera_topic = self.camera_topic
        self._active_camera_topic = self.camera_topic
        self._camera_transport = resolve_camera_transport(
            self._camera_transport_param, self.camera_topic
        )
        self._teach_dist_cache_size = int(self.get_parameter("teach_dist_cache_size").value)
        self._teach_cam_cache_size = int(self.get_parameter("teach_cam_cache_size").value)
        self._teach_lookup_slop_sec = float(self.get_parameter("teach_lookup_slop_sec").value)
        self._record_debug = bool(self.get_parameter("record_debug").value)
        self._record_debug_period_ns = int(
            float(self.get_parameter("record_debug_period_sec").value) * 1e9)
        self._teach_repr_trace = bool(self.get_parameter("teach_repr_trace").value)
        self._teach_repr_log_every = max(
            100, int(self.get_parameter("teach_repr_log_every").value))
        self._teach_feed_watchdog_sec = float(
            self.get_parameter("teach_feed_watchdog_sec").value)

        self._backward_record = False
        self._reset_record_debug_stats()
        self._active_map_dir = ""
        self._teach_msg_filter_subs = []
        self._repr_sub = None
        self._dist_cache = None
        self._cam_cache = None
        self.synced_topics = None
        self._repr_trace_fp = None
        self._teach_feed_watchdog_timer = None
        self._teach_watchdog_last_repr_recv = 0
        self.get_logger().info("Waiting for services to become available...")

        self.distance_reset_cli = self.create_client(SetDist, "teach/set_dist")
        self.align_reset_cli = self.create_client(SetDist, "teach/set_align")

        while not self.distance_reset_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().info("Waiting for teach/set_dist...")

        self.get_logger().info("teach/set_dist is available")

        while not self.align_reset_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().info("Waiting for teach/set_align...")

        self.get_logger().info("teach/set_align is available")

        req = SetDist.Request()
        req.dist = 0.0
        req.map_num = 1
        self.call_service_blocking(self.distance_reset_cli, "teach/set_dist", req)

        req = SetDist.Request()
        req.dist = 0.0
        req.map_num = 1
        self.call_service_blocking(self.align_reset_cli, "teach/set_align", req)

        self.local_align_cli = self.create_client(Alignment, "teach/local_alignment")
        if self.visual_turn:
            while not self.local_align_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for teach/local_alignment...")
            self.get_logger().warn("Local alignment service available for mapmaker")

        # Client for representations' camera rebind service. Best-effort —
        # the mapmaker should still come up if representations isn't ready
        # yet, and we can always call the service lazily on action start.
        self.set_camera_cli = self.create_client(SetCameraTopic, "set_camera_topic")

        self.get_logger().debug("Subscribing to commands")
        self.joy_sub = self.create_subscription(TwistStamped, self.joy_topic, self.joy_cb, NAVIGATION_QOS)

        if self.odom_record_topic:
            self.add_sub = self.create_subscription(Odometry, self.odom_record_topic, self.misc_cb, NAVIGATION_QOS)

        if self.gps_record_topic:
            self.gps_sub = self.create_subscription(
                Odometry, self.gps_record_topic, self._gps_cb, NAVIGATION_QOS)

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
        self._apply_teach_dist_reset(0.0)
        self._log_teach_topic_banner("init")

        self._bag_writer = None
        self._bag_open = False

        self.get_logger().warn("Mapmaker started, awaiting goal")

    def _save_worker_loop(self):
        while True:
            item = self._save_queue.get()
            try:
                if item is None:
                    return
                save_img(*item)
            except Exception as e:
                # Logger may be GC'd during shutdown; guard the logging call.
                try:
                    self.get_logger().error(f"save_img failed in worker: {e}")
                except Exception:
                    print(f"save_img failed in worker: {e}")
            finally:
                self._save_queue.task_done()

    def call_service_blocking(self, client, service_name: str, req, timeout_sec: float = 5.0):
        self.get_logger().info(f"Calling {service_name}...")
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        if not future.done():
            self.get_logger().error(f"{service_name} did not complete in time")
            return None

        result = future.result()
        if result is None:
            self.get_logger().error(f"{service_name} failed")
            return None

        self.get_logger().info(f"Returned from {service_name}")
        return result

    def handle_alignment_async(self, srv_msg):
        if self.align_future is not None and self.align_future.done():
            try:
                resp = self.align_future.result()

                if resp is None or len(resp.histograms) == 0:
                    self.curr_hist = None
                    self.curr_trans = 0.0
                else:
                    hist = resp.histograms[0].data
                    half_size = np.size(hist) / 2.0
                    self.curr_hist = hist
                    self.curr_trans = -float(np.argmax(hist) - (np.size(hist) // 2.0)) / half_size

                #self.get_logger().warn("local_alignment response received")

            except Exception as e:
                self.get_logger().warn(f"Alignment future failed: {e}")
                self.curr_hist = None
                self.curr_trans = 0.0

            self.align_future = None
            self.align_in_progress = False

        if not self.align_in_progress:
            req = Alignment.Request()
            req.input = srv_msg

            #self.get_logger().warn("Sending local_alignment request")
            self.align_future = self.local_align_cli.call_async(req)
            self.align_in_progress = True

    def _teardown_teach_sync(self):
        if self._repr_sub is not None:
            try:
                self.destroy_subscription(self._repr_sub)
            except Exception:
                pass
            self._repr_sub = None
        # message_filters.Subscriber holds an rclpy Subscription in `.sub`.
        for mf_sub in self._teach_msg_filter_subs:
            inner = getattr(mf_sub, "sub", None)
            if inner is not None:
                try:
                    self.destroy_subscription(inner)
                except Exception:
                    pass
        self._teach_msg_filter_subs = []
        self._dist_cache = None
        self._cam_cache = None

    def _camera_sync_qos(self):
        if self._camera_transport == "compressed":
            return CAMERA_SYNC_QOS
        return SYNC_QOS

    def _to_raw_image(self, msg):
        if self._camera_transport == "raw":
            return msg
        decoded, _ = parse_camera_msg(msg, self.bridge)
        return decoded

    def _setup_teach_sync(self):
        self._camera_transport = resolve_camera_transport(
            self._camera_transport_param, self.camera_topic
        )
        cam_msg_type = camera_message_type(self._camera_transport)
        cam_qos = self._camera_sync_qos()

        dist_sub = Subscriber(self, SensorsOutput, "teach/output_dist", qos_profile=SYNC_QOS)
        self._dist_cache = Cache(dist_sub, cache_size=self._teach_dist_cache_size)

        cam_sub = Subscriber(
            self, cam_msg_type, self.camera_topic, qos_profile=cam_qos
        )
        self._cam_cache = Cache(cam_sub, cache_size=self._teach_cam_cache_size)

        self._teach_msg_filter_subs = [dist_sub, cam_sub]

        self._repr_sub = self.create_subscription(
            FeaturesList,
            "live_representation",
            self._repr_cb,
            SYNC_QOS,
        )
        self._log_teach_topic_banner("setup_teach_sync")

    def _fq_topic(self, relative: str) -> str:
        """Fully-qualified topic name for this node's namespace."""
        rel = relative if relative.startswith("/") else relative
        ns = self.get_namespace().rstrip("/")
        if not ns or ns == "/":
            return rel if rel.startswith("/") else f"/{rel}"
        if rel.startswith("/"):
            return rel
        return f"{ns}/{rel}"

    def _teach_feed_topics(self) -> dict:
        return {
            "namespace": self.get_namespace(),
            "live_representation": self._fq_topic("live_representation"),
            "teach_output_dist": self._fq_topic("teach/output_dist"),
            "camera": self.camera_topic,
            "set_camera_service": self._fq_topic("set_camera_topic"),
            "representations_hint": self._fq_topic("representations"),
        }

    def _repr_sub_publisher_count(self) -> int:
        if self._repr_sub is None:
            return -1
        try:
            return self._repr_sub.get_publisher_count()
        except Exception:
            return -1

    def _log_teach_topic_banner(self, where: str) -> None:
        t = self._teach_feed_topics()
        repr_pubs = self._repr_sub_publisher_count()
        peer = "pfvtr_map" if "pfvtr_map" in t["namespace"] else "pfvtr"
        other = "pfvtr" if peer == "pfvtr_map" else "pfvtr_map"
        self.get_logger().warn(
            f"[teach_topics/{where}] node={self.get_fully_qualified_name()} "
            f"namespace={t['namespace']!r} | "
            f"SUB repr={t['live_representation']} (pubs={repr_pubs}) | "
            f"SUB dist={t['teach_output_dist']} | "
            f"SUB camera={t['camera']} ({self._camera_transport}) | "
            f"SVC set_camera={t['set_camera_service']} | "
            f"rear={self.camera_back_topic!r} default_front={self._default_camera_topic!r} | "
            f"CHECK: rear map ({peer}) needs hz on {t['live_representation']} "
            f"NOT /{other}/live_representation"
        )
        if repr_pubs == 0:
            self.get_logger().error(
                f"[teach_topics/{where}] NO PUBLISHER on {t['live_representation']} — "
                f"mapmaker will never receive live_representation. "
                f"Is {t['namespace']}/representations running? "
                f"(ros2 topic hz {t['live_representation']})"
            )

    def _open_repr_trace(self, map_name: str) -> None:
        self._close_repr_trace()
        if not self._teach_repr_trace:
            return
        try:
            trace_path = _map_path(map_name, "teach_repr_trace.log")
            self._repr_trace_fp = open(trace_path, "a", encoding="utf-8")
            t = self._teach_feed_topics()
            self._repr_trace_fp.write(
                f"# teach start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"map={map_name!r} backward={self._backward_record}\n"
                f"# repr={t['live_representation']} dist={t['teach_output_dist']} "
                f"camera={t['camera']}\n"
            )
            self._repr_trace_fp.flush()
            self.get_logger().warn(f"[teach_repr_trace] writing -> {trace_path}")
        except Exception as e:
            self.get_logger().error(f"[teach_repr_trace] cannot open trace file: {e}")
            self._repr_trace_fp = None

    def _close_repr_trace(self) -> None:
        if self._repr_trace_fp is not None:
            try:
                self._repr_trace_fp.write(
                    f"# teach end {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"repr_recv={self._teach_sync_stats.get('repr_recv', 0)} "
                    f"sync_ok={self._teach_sync_stats.get('sync_ok', 0)}\n"
                )
                self._repr_trace_fp.flush()
                self._repr_trace_fp.close()
            except Exception:
                pass
            self._repr_trace_fp = None

    def _repr_trace_write(self, line: str) -> None:
        if self._repr_trace_fp is None:
            return
        try:
            self._repr_trace_fp.write(line + "\n")
            self._repr_trace_fp.flush()
        except Exception as e:
            self.get_logger().error(f"[teach_repr_trace] write failed: {e}")

    def _start_teach_feed_watchdog(self) -> None:
        self._stop_teach_feed_watchdog()
        self._teach_watchdog_last_repr_recv = 0
        period = max(0.5, self._teach_feed_watchdog_sec)
        self._teach_feed_watchdog_timer = self.create_timer(
            period, self._teach_feed_watchdog_cb)

    def _stop_teach_feed_watchdog(self) -> None:
        if self._teach_feed_watchdog_timer is not None:
            try:
                self._teach_feed_watchdog_timer.cancel()
                self.destroy_timer(self._teach_feed_watchdog_timer)
            except Exception:
                pass
            self._teach_feed_watchdog_timer = None

    def _teach_feed_watchdog_cb(self) -> None:
        if not self.isMapping:
            return
        recv = self._teach_sync_stats['repr_recv']
        t = self._teach_feed_topics()
        repr_pubs = self._repr_sub_publisher_count()
        dist_n, cam_n = self._teach_cache_sizes()
        if recv == self._teach_watchdog_last_repr_recv:
            self.get_logger().error(
                f"[teach_feed_watchdog] NO live_representation at mapmaker "
                f"since last {self._teach_feed_watchdog_sec:.1f}s | "
                f"expecting {t['live_representation']} pubs={repr_pubs} | "
                f"repr.recv={recv} sync_ok={self._teach_sync_stats['sync_ok']} | "
                f"dist_cache={dist_n} cam_cache={cam_n} camera={t['camera']!r} | "
                f"run: ros2 topic hz {t['live_representation']}"
            )
        self._teach_watchdog_last_repr_recv = recv

    def _warn_teach_lookup_throttled(self, message: str):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_teach_lookup_warn_ns < 2_000_000_000:
            return
        self._last_teach_lookup_warn_ns = now_ns
        self.get_logger().warn(message)

    def _teach_cache_sizes(self) -> tuple:
        dist_n = len(self._dist_cache.cache_msgs) if self._dist_cache is not None else 0
        cam_n = len(self._cam_cache.cache_msgs) if self._cam_cache is not None else 0
        return dist_n, cam_n

    def _apply_teach_dist_reset(self, dist: float) -> None:
        # Local boundary after teach/set_dist: the service resets the fusion
        # estimator, but mapmaker keeps its own latched state (distance cache,
        # self.dist, last_action_dist) that must be cleared in the same beat.
        if self._dist_cache is not None:
            self._dist_cache.cache_msgs = []
            self._dist_cache.cache_times = []
        self.dist = float(dist)
        self.last_action_dist = float(dist)
        self.nextStep = float(dist)
        self._startup_skip_count = 3
        self._teach_bag_armed = False

    def _teach_sync_fail(self, kind: str, detail: str) -> None:
        self._teach_sync_stats[kind] += 1
        self._teach_sync_stats['last_fail'] = f"{kind}: {detail}"
        self._warn_teach_lookup_throttled(f"Teach sync FAIL [{kind}] {detail}")

    def _repr_cb(self, repr_msg: FeaturesList):
        self._teach_sync_stats['repr_recv'] += 1
        n = self._teach_sync_stats['repr_recv']
        dist_n, cam_n = self._teach_cache_sizes()
        stamp_s = None
        if hasattr(repr_msg, "header") and repr_msg.header is not None:
            stamp_s = (
                repr_msg.header.stamp.sec
                + repr_msg.header.stamp.nanosec / 1e9
            )

        trace_line = (
            f"repr#{n} stamp={stamp_s} mapping={self.isMapping} "
            f"dist_cache={dist_n} cam_cache={cam_n} pf_dist={self.dist:.3f}"
        )
        self._repr_trace_write(trace_line)

        if (self._teach_repr_log_every == 1
                or n <= 10
                or n % self._teach_repr_log_every == 0
                or self.isMapping is not False):
            self.get_logger().info(
                f"[teach_repr] {trace_line} topic={self._fq_topic('live_representation')}"
            )

        if self._record_debug and not self._first_repr_recv:
            self._first_repr_recv = True
            self._record_debug_log(
                f'first live_representation #{n} dist_cache={dist_n} cam_cache={cam_n} '
                f'camera={self.camera_topic!r} slop={self._teach_lookup_slop_sec:.2f}s '
                f'topic={self._fq_topic("live_representation")}'
            )

        if not hasattr(repr_msg, "header") or repr_msg.header is None:
            self._teach_sync_fail('fail_no_header', 'live_representation missing header')
            return

        stamp = Time.from_msg(repr_msg.header.stamp)
        slop = Duration(seconds=self._teach_lookup_slop_sec)

        dist_msg = _lookup_nearest(self._dist_cache, stamp, slop)
        if dist_msg is None:
            miss = _lookup_miss_info(self._dist_cache, stamp, slop)
            dist_n, cam_n = self._teach_cache_sizes()
            self._teach_sync_fail(
                'fail_no_dist',
                f"teach/output_dist — {miss['reason']} "
                f"(dist_cache={dist_n} cam_cache={cam_n} "
                f"repr_stamp={stamp.nanoseconds / 1e9:.3f})",
            )
            return

        img_msg = _lookup_nearest(self._cam_cache, stamp, slop)
        if img_msg is None:
            miss = _lookup_miss_info(self._cam_cache, stamp, slop)
            dist_n, cam_n = self._teach_cache_sizes()
            self._teach_sync_fail(
                'fail_no_cam',
                f"camera {self.camera_topic!r} — {miss['reason']} "
                f"(dist_cache={dist_n} cam_cache={cam_n} "
                f"repr_stamp={stamp.nanoseconds / 1e9:.3f})",
            )
            return

        img_msg = self._to_raw_image(img_msg)
        if img_msg is None:
            self._teach_sync_fail(
                'fail_decode',
                f"camera {self.camera_topic!r} transport={self._camera_transport!r}",
            )
            return

        self._teach_sync_stats['sync_ok'] += 1
        self._repr_trace_write(
            f"  -> sync_ok#{self._teach_sync_stats['sync_ok']} "
            f"dist={float(dist_msg.output):.3f} mapping={self.isMapping}"
        )
        self.distance_img_cb(repr_msg, dist_msg, img_msg)

    async def _request_representations_camera(self, topic: str) -> bool:
        if not self.set_camera_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "set_camera_topic service not available — representations may not rebind"
            )
            return False
        req = SetCameraTopic.Request()
        req.topic = topic
        future = self.set_camera_cli.call_async(req)
        await future
        resp = future.result()
        if resp is None:
            self.get_logger().warn("set_camera_topic call returned no response")
            return False
        return bool(resp.success)

    async def _rebind_camera(self, topic: str):
        """Switch both representations' and mapmaker's camera subscription."""
        if topic == self._active_camera_topic:
            self.get_logger().info(
                f"Camera already on '{topic}' — skipping rebind"
            )
            return
        ok = await self._request_representations_camera(topic)
        if not ok:
            self._bag_error_banner(
                "REPRESENTATIONS CAMERA REBIND FAILED",
                f"requested={topic!r} service={self._fq_topic('set_camera_topic')}",
                "Mapmaker will subscribe to the camera locally but "
                f"{self._fq_topic('representations')} may still be on the wrong camera — "
                "live_representation may be stale or absent.",
            )
        self.camera_topic = topic
        self._teardown_teach_sync()
        self._setup_teach_sync()
        self._active_camera_topic = topic
        self._camera_transport = resolve_camera_transport(
            self._camera_transport_param, self.camera_topic
        )
        self.get_logger().warn(
            f"Mapmaker camera rebound to '{topic}' ({self._camera_transport}) "
            f"repr_rebind_ok={ok}"
        )

    def _setup_repeat_sync(self):
        self._camera_transport = resolve_camera_transport(
            self._camera_transport_param, self.camera_topic
        )
        cam_msg_type = camera_message_type(self._camera_transport)
        cam_qos = self._camera_sync_qos()

        repr_sub = Subscriber(self, FeaturesList, "live_representation", qos_profile=SYNC_QOS)
        cam_sub = Subscriber(
            self, cam_msg_type, self.camera_topic, qos_profile=cam_qos
        )
        distance_sub = Subscriber(self, SensorsOutput, "repeat/output_dist", qos_profile=SYNC_QOS)
        align_sub = Subscriber(self, SensorsOutput, "repeat/output_align", qos_profile=SYNC_QOS)

        self.synced_topics = ApproximateTimeSynchronizer(
            [repr_sub, distance_sub, align_sub, cam_sub],
            queue_size=20,
            slop=0.5
        )
        self.synced_topics.registerCallback(self.distance_wrapper_cb)


    def misc_cb(self, msg: Odometry):
        # Always cache the most recent odometry so the action_cb can record
        # the robot's pose at the *moment the mapping goal arrives*, not at
        # the first action_dist_step after motion starts.  Dropping the
        # `isMapping` gate is safe — `lastOdom` is only consumed inside the
        # twist-callback bag-write path, which itself runs only while a bag
        # is open.
        self.lastOdom = msg

    def _gps_cb(self, msg: Odometry):
        # Same caching rationale as misc_cb — consumed only while a bag is open.
        self.lastGpsOdom = msg

    def distance_wrapper_cb(
        self,
        repr_msg: FeaturesList,
        dist_msg: SensorsOutput,
        align_msg: SensorsOutput,
        img,
    ):
        self.curr_alignment = align_msg.output
        img = self._to_raw_image(img)
        if img is None:
            return
        self.distance_img_cb(repr_msg, dist_msg, img)

    def distance_img_cb(self, repr_msg: FeaturesList, dist_msg: SensorsOutput, img: Image):
        dist = float(dist_msg.output)
        #self.get_logger().warn(f"SYNC CB fired | isMapping={self.isMapping} | dist={dist}")

        feat0 = repr_msg.data[0]
        values = np.array(feat0.values).reshape(feat0.shape)
        self.img_features = [values, feat0.descriptors]

        self.img_msg = img
        self.header = repr_msg.header
        dist = float(dist_msg.output)
        if not self.isMapping:
            self._teach_sync_stats['skip_not_mapping'] += 1
            return

        # Flush repr frames still in flight from before isMapping flipped.
        # Their distance lookup can reflect pre teach/set_dist state.
        if self._startup_skip_count > 0:
            self._teach_sync_stats['skip_startup'] += 1
            self._startup_skip_count -= 1
            if self._startup_skip_count == 0:
                self._teach_bag_armed = True
            return

        self.dist = dist

        # obtain displacement between prev and new image
        if self.visual_turn and self.last_img_features is not None and dist:
            srv_msg = SensorsInput()
            srv_msg.map_features = [numpy_to_feature(self.last_img_features)]
            srv_msg.live_features = [numpy_to_feature(self.img_features)]

            self.handle_alignment_async(srv_msg)
        else:
            self.curr_trans = 0.0
            self.curr_hist = None

        # eventually save the image if conditions fulfilled
        if self.target_distances is not None and self.curr_hist is not None:
            desired_idx = int(np.argmin(abs(dist - np.array(self.target_distances))))
            self.last_img_features = self.img_features
            if self.collected_distances[desired_idx] == 0 and self.target_distances[desired_idx] <= dist:
                self.collected_distances[desired_idx] = 1
                self._save_queue.put((
                    self.img_features, self.img_msg, self.header, self.mapName, dist,
                    self.curr_hist, self.curr_alignment, self.source_map, self.save_imgs, self.bridge
                ))
                self._note_kp_saved(dist)
                self.get_logger().info(f"Saved waypoint: {dist}, {self.curr_trans}")

        # save after fixed distance OR visual turn threshold
        if self.target_distances is None and (dist > self.nextStep or abs(self.curr_trans) > self.max_trans):
            self.nextStep = dist + self.mapStep
            self.last_img_features = self.img_features
            self._save_queue.put((
                self.img_features, self.img_msg, self.header, self.mapName, dist,
                self.curr_hist, self.curr_alignment, self.source_map, self.save_imgs, self.bridge
            ))
            self._note_kp_saved(dist)
            self.get_logger().info(f"Saved waypoint: {dist}, {self.curr_trans}")

        if self.last_img_features is None:
            self.last_img_features = self.img_features

        self._maybe_log_record_status('sync')
        self.checkShutdown()

    def _reset_record_debug_stats(self) -> None:
        self._joy_stats = {
            'recv': 0,
            'saved': 0,
            'skip_no_bag': 0,
            'skip_dist_low': 0,
            'skip_dist_step': 0,
        }
        self._teach_sync_stats = {
            'repr_recv': 0,
            'sync_ok': 0,
            'fail_no_header': 0,
            'fail_no_dist': 0,
            'fail_no_cam': 0,
            'fail_decode': 0,
            'skip_not_mapping': 0,
            'skip_startup': 0,
            'kp_queued': 0,
            'last_fail': '',
        }
        self._last_kp_dist = 0.0
        self._max_kp_dist = 0.0
        self._max_action_dist = 0.0
        self._last_action_vx = 0.0
        self._last_action_wz = 0.0
        self._last_joy_vx = 0.0
        self._last_joy_wz = 0.0
        self._record_debug_last_log_ns = 0
        self._first_joy_saved = False
        self._first_joy_recv = False
        self._first_repr_recv = False
        self._first_saved_action_dist = None
        self._min_kp_dist = float('inf')
        self._warned_no_bag = False
        self._warned_dist_stuck = False
        self._bag_write_failures = 0

    @staticmethod
    def _error_banner(headline: str, *lines: str) -> str:
        body = "".join(f"\n  {line}" for line in lines)
        return (
            "\n" + "!" * 72 +
            f"\n  {headline}" +
            body +
            "\n" + "!" * 72
        )

    def _bag_error_banner(self, headline: str, *lines: str) -> None:
        self.get_logger().error(self._error_banner(headline, *lines))

    def _bag_write(self, topic: str, msg, stamp_ns: int) -> bool:
        if self._bag_writer is None:
            self._bag_error_banner(
                "BAG WRITE FAILED — bag writer is not open",
                f"topic={topic}",
                f"map={self.mapName!r}",
                "Recording is active but nothing can be saved to the bag.",
            )
            return False
        try:
            self._bag_writer.write(topic, serialize_message(msg), stamp_ns)
            return True
        except Exception as e:
            self._bag_write_failures += 1
            self._bag_error_banner(
                f"BAG WRITE FAILED — could not write {topic}",
                f"map={self.mapName!r}",
                f"error={e}",
            )
            return False

    def _record_debug_log(self, msg: str) -> None:
        if self._record_debug:
            self.get_logger().warn(f'[record] {msg}')

    def _note_kp_saved(self, dist: float) -> None:
        self._teach_sync_stats['kp_queued'] += 1
        self._last_kp_dist = float(dist)
        self._max_kp_dist = max(self._max_kp_dist, self._last_kp_dist)
        self._min_kp_dist = min(self._min_kp_dist, self._last_kp_dist)

    def _maybe_log_record_status(self, where: str) -> None:
        if not self._record_debug or not self.isMapping:
            return
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._record_debug_last_log_ns) < self._record_debug_period_ns:
            return
        self._record_debug_last_log_ns = now_ns
        pubs = 0
        try:
            pubs = self.joy_sub.get_publisher_count()
        except Exception:
            pass
        s = self._joy_stats
        ts = self._teach_sync_stats
        dist_n, cam_n = self._teach_cache_sizes()
        self.get_logger().warn(
            f'[record/{where}] map={self.mapName!r} dist={self.dist:.2f}m '
            f'last_act={self.last_action_dist:.2f}m last_kp={self._last_kp_dist:.2f}m '
            f'gap_act={self.dist - self.last_action_dist:.2f}m '
            f'gap_kp={self.dist - self._last_kp_dist:.2f}m | '
            f'cmd_vel@{self.joy_topic} pubs={pubs} '
            f'joy.recv={s["recv"]} saved={s["saved"]} '
            f'skip(no_bag={s["skip_no_bag"]} dist_low={s["skip_dist_low"]} '
            f'dist_step={s["skip_dist_step"]}) '
            f'last_joy vx={self._last_joy_vx:+.3f} wz={self._last_joy_wz:+.3f} '
            f'last_saved vx={self._last_action_vx:+.3f} wz={self._last_action_wz:+.3f} | '
            f'teach repr.recv={ts["repr_recv"]} sync_ok={ts["sync_ok"]} '
            f'kp_queued={ts["kp_queued"]} dist_cache={dist_n} cam_cache={cam_n} | '
            f'teach_fail(hdr={ts["fail_no_header"]} dist={ts["fail_no_dist"]} '
            f'cam={ts["fail_no_cam"]} decode={ts["fail_decode"]} '
            f'skip_startup={ts["skip_startup"]})'
        )

    def _dump_record_summary(self, phase: str) -> None:
        s = self._joy_stats
        act_gap = float(self.dist) - float(self.last_action_dist)
        kp_gap = float(self.dist) - float(self._last_kp_dist)
        pubs = 0
        try:
            pubs = self.joy_sub.get_publisher_count()
        except Exception:
            pass
        self.get_logger().warn(
            f'[record/{phase}] SUMMARY map={self.mapName!r} '
            f'pf_dist={self.dist:.3f}m max_kp={self._max_kp_dist:.3f}m '
            f'max_action={self._max_action_dist:.3f}m | '
            f'last_action={self.last_action_dist:.3f}m '
            f'last_kp={self._last_kp_dist:.3f}m | '
            f'END GAP action={act_gap:+.3f}m keypoint={kp_gap:+.3f}m | '
            f'cmd_vel@{self.joy_topic} pubs={pubs} | '
            f'joy.recv={s["recv"]} saved={s["saved"]} '
            f'skip(no_bag={s["skip_no_bag"]} dist_low={s["skip_dist_low"]} '
            f'dist_step={s["skip_dist_step"]}) | '
            f'last_saved vx={self._last_action_vx:+.3f} wz={self._last_action_wz:+.3f}'
        )
        self._dump_teach_sync_summary(phase)
        if (
            self._first_saved_action_dist is not None
            and self._min_kp_dist < float('inf')
            and self._first_saved_action_dist - self._min_kp_dist > 0.5
        ):
            self._bag_error_banner(
                "TEACH BAG MISALIGNED — first action far from first waypoint",
                f"map={self.mapName!r} first_kp={self._min_kp_dist:.3f}m "
                f"first_action={self._first_saved_action_dist:.3f}m",
                "Replay will fail unless the map is re-taught from a clean reset.",
            )
        if act_gap > 0.05:
            self.get_logger().error(
                f'[record/{phase}] ACTIONS STOPPED {act_gap:.2f}m BEFORE PF DISTANCE '
                f'(keypoints may continue; replay will miss this tail)'
            )
        if s['recv'] == 0:
            self.get_logger().error(
                f'[record/{phase}] NO cmd_vel RECEIVED on {self.joy_topic} '
                f'during teach — check twist_mux / topic remap'
            )
        elif s['recv'] > 0 and s['saved'] == 0:
            self._bag_error_banner(
                "RECORDING FINISHED WITH EMPTY BAG — zero cmd_vel actions saved",
                f"map={self.mapName!r} pf_dist={self.dist:.3f}m",
                f"cmd_vel received {s['recv']}x but bag has no /recorded_actions",
                f"skip_dist_low={s['skip_dist_low']} skip_dist_step={s['skip_dist_step']} "
                f"skip_no_bag={s['skip_no_bag']}",
                "Check teach/output_dist (PF distance stuck at 0?) and cmd_vel topic.",
            )
        elif s['saved'] == 0:
            self._bag_error_banner(
                "RECORDING FINISHED WITH EMPTY BAG — no /recorded_actions written",
                f"map={self.mapName!r} pf_dist={self.dist:.3f}m",
                "No cmd_vel was received during teach — bag contains no motion commands.",
            )
        if self._bag_write_failures > 0:
            self._bag_error_banner(
                "RECORDING HAD BAG WRITE FAILURES",
                f"map={self.mapName!r} failed_writes={self._bag_write_failures}",
                "The saved bag may be incomplete — inspect maps/<name>/bag/.",
            )

    def _dump_teach_sync_summary(self, phase: str) -> None:
        ts = self._teach_sync_stats
        dist_n, cam_n = self._teach_cache_sizes()
        self.get_logger().warn(
            f'[teach_sync/{phase}] SUMMARY repr.recv={ts["repr_recv"]} '
            f'sync_ok={ts["sync_ok"]} kp_queued={ts["kp_queued"]} '
            f'max_kp_dist={self._max_kp_dist:.3f}m pf_dist={self.dist:.3f}m | '
            f'fail(hdr={ts["fail_no_header"]} dist={ts["fail_no_dist"]} '
            f'cam={ts["fail_no_cam"]} decode={ts["fail_decode"]}) '
            f'skip(not_mapping={ts["skip_not_mapping"]} startup={ts["skip_startup"]}) | '
            f'feeds dist_cache={dist_n} cam_cache={cam_n} '
            f'camera={self.camera_topic!r} slop={self._teach_lookup_slop_sec:.2f}s'
        )
        if ts['repr_recv'] == 0:
            t = self._teach_feed_topics()
            self._bag_error_banner(
                "TEACH SYNC DEAD — zero live_representation received",
                f"map={self.mapName!r} camera={self.camera_topic!r}",
                f"namespace={t['namespace']!r} expected_repr={t['live_representation']}",
                f"repr_pubs_at_stop={self._repr_sub_publisher_count()}",
                "Mapmaker never got a repr frame — no .npy keypoints or PF dist updates.",
                f"Check: ros2 topic hz {t['live_representation']} "
                f"(NOT /pfvtr/live_representation if this is pfvtr_map).",
                f"Check {t['namespace']}/representations is running and bound to {t['camera']}.",
            )
        elif ts['sync_ok'] == 0:
            last = ts['last_fail'] or '(no detail)'
            self._bag_error_banner(
                "TEACH SYNC NEVER SUCCEEDED — repr arrived but dist/cam lookup always failed",
                f"map={self.mapName!r} repr.recv={ts['repr_recv']}",
                f"fail dist={ts['fail_no_dist']} cam={ts['fail_no_cam']} "
                f"decode={ts['fail_decode']} hdr={ts['fail_no_header']}",
                f"last_fail={last}",
                f"feeds dist_cache={dist_n} cam_cache={cam_n}",
                "Empty dist_cache -> teach/output_dist silent (sensors/odom).",
                "Empty cam_cache -> camera topic silent or QoS mismatch.",
                "stamp skew -> raise teach_lookup_slop_sec or fix clock sync.",
            )
        elif ts['kp_queued'] == 0 and self._max_kp_dist <= 0.0:
            self._bag_error_banner(
                "TEACH SYNC OK BUT NO KEYPOINTS QUEUED",
                f"map={self.mapName!r} sync_ok={ts['sync_ok']} pf_dist={self.dist:.3f}m",
                f"skip_startup={ts['skip_startup']} map_step={self.mapStep:.2f}m",
                "Repr+cam+dist aligned but robot may not have moved map_step yet,",
                "or distance stayed at 0 (teach/output_dist / odom not advancing).",
            )

    def joy_cb(self, msg: TwistStamped):
        if self.isMapping:
            if not self._teach_bag_armed:
                return
            self._joy_stats['recv'] += 1
            self._last_joy_vx = float(msg.twist.linear.x)
            self._last_joy_wz = float(msg.twist.angular.z)
            if self._record_debug and not self._first_joy_recv:
                self._first_joy_recv = True
                pubs = 0
                try:
                    pubs = self.joy_sub.get_publisher_count()
                except Exception:
                    pass
                self._record_debug_log(
                    f'first cmd_vel while mapping on {self.joy_topic} '
                    f'(pubs={pubs}) dist={self.dist:.3f}m '
                    f'vx={self._last_joy_vx:+.3f} wz={self._last_joy_wz:+.3f}'
                )
            if self._bag_writer is None:
                self._joy_stats['skip_no_bag'] += 1
                if not self._warned_no_bag:
                    self._warned_no_bag = True
                    self._bag_error_banner(
                        "RECORDING ACTIVE BUT BAG IS NOT OPEN — cmd_vel will not be saved",
                        f"map={self.mapName!r} topic={self.joy_topic}",
                    )
                return
            if self.dist < 0.0:
                self._joy_stats['skip_dist_low'] += 1
                if not self._warned_dist_stuck:
                    self._warned_dist_stuck = True
                    self._bag_error_banner(
                        "RECORDING ACTIVE BUT TEACH DISTANCE IS ZERO — bag writes blocked",
                        f"map={self.mapName!r} pf_dist={self.dist:.3f}m",
                        "Waiting for teach/output_dist to advance before saving cmd_vel/odom.",
                        "No /recorded_actions or /recorded_odometry will be written until distance > 0.01 m.",
                    )
                return

            dist_delta = self.dist - self.last_action_dist
            if dist_delta >= self.action_dist_step:
                save_msg = DistancedTwist()
                save_msg.twist = msg.twist
                save_msg.distance = float(self.dist)

                now_ns = self.get_clock().now().nanoseconds
                if not self._bag_write("/recorded_actions", save_msg, now_ns):
                    return
                self.last_action_dist = self.dist
                self._max_action_dist = max(self._max_action_dist, self.last_action_dist)
                self._last_action_vx = float(msg.twist.linear.x)
                self._last_action_wz = float(msg.twist.angular.z)
                self._joy_stats['saved'] += 1
                if self._first_saved_action_dist is None:
                    self._first_saved_action_dist = float(self.dist)
                if self._record_debug and not self._first_joy_saved:
                    self._first_joy_saved = True
                    self._record_debug_log(
                        f'first action saved at dist={self.dist:.3f}m '
                        f'vx={self._last_action_vx:+.3f} wz={self._last_action_wz:+.3f}'
                    )

                if self.lastOdom is not None:
                    if not self._bag_write("/recorded_odometry", self.lastOdom, now_ns):
                        self.get_logger().error(
                            f"Action saved at dist={self.dist:.3f}m but "
                            "/recorded_odometry write failed — bag may be mismatched"
                        )

                if self.lastGpsOdom is not None:
                    self._bag_write("/recorded_gps", self.lastGpsOdom, now_ns)
            else:
                self._joy_stats['skip_dist_step'] += 1
            self._maybe_log_record_status('joy')

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

    def _bag_open_for_map(self, map_name: str) -> bool:
        bag_dir = _map_path(map_name, "bag")
        t0 = time.perf_counter()

        storage_options = rosbag2_py.StorageOptions(
            uri=bag_dir,
            storage_id="mcap"
        )

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )

        try:
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
            if self.gps_record_topic:
                self._bag_writer.create_topic(rosbag2_py.TopicMetadata(
                    id=0,
                    name="/recorded_gps",
                    type="nav_msgs/msg/Odometry",
                    serialization_format="cdr"
                ))
        except Exception as e:
            elapsed_s = time.perf_counter() - t0
            self._bag_writer = None
            self._bag_open = False
            self._bag_error_banner(
                "BAG OPEN FAILED — recording cannot save cmd_vel or odometry",
                f"map={map_name!r}",
                f"bag_dir={bag_dir}",
                f"open_time={elapsed_s:.3f}s",
                f"error={e}",
            )
            return False

        elapsed_s = time.perf_counter() - t0
        self._bag_open_elapsed_s = elapsed_s
        self.get_logger().warn(
            f"[record] bag open for cmd_vel/odom writes took {elapsed_s:.3f}s "
            f"(map={map_name!r} dir={bag_dir})"
        )
        self._bag_open = True
        return True

    def _bag_close(self):
        # SequentialWriter closes on destruction
        self._bag_writer = None
        self._bag_open = False

    def _postprocess_reverse_map(self, map_dir: str):
        """Rename waypoints and rewrite action bag for a map recorded backward.

        After this runs, filename 0 corresponds to the *end* of the recorded
        drive (point B) and filename max_dist corresponds to the start (A).
        The action bag is rewritten with distances rebased and angular.z
        negated, so forward replay with the front camera retraces the path
        from B back to A after a physical 180° turn.
        """
        self.get_logger().warn(f"Post-processing reverse map at {map_dir}")

        try:
            distances = get_map_dists(map_dir).tolist()
        except Exception as e:
            self.get_logger().warn(f"No .npy files found; skipping reversal: {e}")
            return
        if not distances:
            return
        max_dist = float(distances[-1])
        if max_dist <= 0.0:
            self.get_logger().warn(
                f"Non-positive max_dist ({max_dist}); skipping reversal"
            )
            return

        # Two-phase rename avoids collisions when new distance equals an
        # existing old distance (e.g. symmetric midpoints).
        tmp_suffix = ".revtmp"
        for d in distances:
            for ext in (".npy", ".jpg"):
                old = os.path.join(map_dir, f"{d}{ext}")
                if os.path.exists(old):
                    os.rename(old, old + tmp_suffix)
        for d in distances:
            new_d = max_dist - float(d)
            for ext in (".npy", ".jpg"):
                tmp_path = os.path.join(map_dir, f"{d}{ext}{tmp_suffix}")
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, os.path.join(map_dir, f"{new_d}{ext}"))

        bag_dir = os.path.join(map_dir, "bag")
        if os.path.isdir(bag_dir):
            self._rewrite_bag_reversed(bag_dir, max_dist)
        else:
            self.get_logger().warn(f"No bag at {bag_dir}; skipping bag reversal")

        try:
            with open(os.path.join(map_dir, "params"), "a") as f:
                f.write("recordedDirection: backward\n")
        except Exception as e:
            self.get_logger().warn(f"Could not append direction to params: {e}")

    def _rewrite_bag_reversed(self, bag_dir: str, max_dist: float):
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_dir, storage_id="mcap"),
            converter_options,
        )
        topics_info = reader.get_all_topics_and_types()
        type_by_name = {t.name: t.type for t in topics_info}

        messages = []  # (orig_ns, topic, data)
        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            messages.append((t_ns, topic, data))
        del reader  # SequentialReader closes on destruction

        # Group messages into per-waypoint frames by original timestamp: the
        # mapmaker writes /recorded_actions and /recorded_odometry with the
        # SAME now_ns in each joy_cb, so a shared timestamp identifies one
        # waypoint. Keeping action and its odometry paired (same output stamp)
        # is what lets the repeater's trajectory mode re-associate poses with
        # distances after reversal.
        frames = {}  # orig_ns -> {"action": data, "odom": data, "gps": data, "other": [(topic, data)]}
        order = []   # orig_ns in first-seen order
        for (t_ns, topic, data) in messages:
            if t_ns not in frames:
                frames[t_ns] = {"action": None, "odom": None, "gps": None, "other": []}
                order.append(t_ns)
            if topic == "/recorded_actions":
                frames[t_ns]["action"] = data
            elif topic == "/recorded_odometry":
                frames[t_ns]["odom"] = data
            elif topic == "/recorded_gps":
                frames[t_ns]["gps"] = data
            else:
                frames[t_ns]["other"].append((topic, data))

        # Reverse frame order so distance runs ascending from point B after the
        # per-frame distance rebase.
        order.sort()
        order.reverse()

        tmp_dir = bag_dir + ".rev"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=tmp_dir, storage_id="mcap"),
            converter_options,
        )
        for name, typ in type_by_name.items():
            writer.create_topic(rosbag2_py.TopicMetadata(
                id=0,
                name=name,
                type=typ,
                serialization_format="cdr",
            ))

        # Monotonically-increasing synthetic timestamps; the repeater matches
        # actions by `msg.distance`, and pairs odometry with the action sharing
        # the same stamp, so exact original timing isn't required — only that
        # action and its odometry keep a common stamp.
        base_ns = self.get_clock().now().nanoseconds
        step_ns = 1_000_000  # 1 ms
        for i, t_ns in enumerate(order):
            frame = frames[t_ns]
            stamp = base_ns + i * step_ns

            # Action: rebase distance to ascend from B, negate angular.z.
            if frame["action"] is not None:
                data = frame["action"]
                try:
                    msg = deserialize_message(data, DistancedTwist)
                    msg.distance = float(max_dist) - float(msg.distance)
                    msg.twist.angular.z = float(-msg.twist.angular.z)
                    # linear.x is intentionally left untouched — it was
                    # positive during forward teach and must remain positive
                    # for the forward repeat after the robot turns around.
                    data = serialize_message(msg)
                except Exception as e:
                    self.get_logger().warn(
                        f"Failed to transform /recorded_actions entry: {e}"
                    )
                writer.write("/recorded_actions", data, stamp)

            # Odometry: keep position, rotate heading by 180° — the robot
            # retraces the same spatial path facing the opposite way after the
            # physical turn. Written with the action's stamp to preserve the
            # pairing the repeater's trajectory mode relies on.
            if frame["odom"] is not None:
                data = frame["odom"]
                try:
                    odom = deserialize_message(data, Odometry)
                    q = odom.pose.pose.orientation
                    yaw = math.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                    ) + math.pi
                    q.x = 0.0
                    q.y = 0.0
                    q.z = math.sin(yaw / 2.0)
                    q.w = math.cos(yaw / 2.0)
                    data = serialize_message(odom)
                except Exception as e:
                    self.get_logger().warn(
                        f"Failed to transform /recorded_odometry entry: {e}"
                    )
                writer.write("/recorded_odometry", data, stamp)

            if frame["gps"] is not None:
                data = frame["gps"]
                try:
                    gps = deserialize_message(data, Odometry)
                    q = gps.pose.pose.orientation
                    yaw = math.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                    ) + math.pi
                    q.x = 0.0
                    q.y = 0.0
                    q.z = math.sin(yaw / 2.0)
                    q.w = math.cos(yaw / 2.0)
                    data = serialize_message(gps)
                except Exception as e:
                    self.get_logger().warn(
                        f"Failed to transform /recorded_gps entry: {e}"
                    )
                writer.write("/recorded_gps", data, stamp)

            # Any other topics: pass through unchanged at this stamp.
            for (topic, data) in frame["other"]:
                writer.write(topic, data, stamp)

        del writer

        shutil.rmtree(bag_dir)
        os.rename(tmp_dir, bag_dir)
        self.get_logger().warn(f"Reversed bag written to {bag_dir}")

    async def action_cb(self, goal_handle):
        self._active_goal_handle = goal_handle
        goal = goal_handle.request
        result = MapMaker.Result()

        if goal.source_map != "":
            self.target_distances = []
            self.source_map = goal.source_map
            self.target_distances = get_map_dists(_map_path(self.source_map))
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

            self._backward_record = bool(goal.record_backward)

            # Backward mapping requires a configured rear camera. If the
            # user requested it without one available, reject the goal
            # instead of silently falling back to the front camera.
            if self._backward_record and not self.camera_back_topic:
                self.get_logger().error(
                    "record_backward=true but `camera_back_topic` parameter "
                    "is empty — aborting goal. Set camera_back_topic in the "
                    "launch file to enable backward mapping."
                )
                result.success = False
                goal_handle.abort()
                return result

            requested_cam = (
                self.camera_back_topic
                if self._backward_record
                else self._default_camera_topic
            )
            # Rebind before isMapping flips so the sync stream is already on
            # the right topic by the time frames start flowing in.
            await self._rebind_camera(requested_cam)
            self._active_map_dir = goal.map_name

            req = SetDist.Request()
            req.dist = 0.0
            req.map_num = 1
            # We're inside an async action callback dispatched by the executor,
            # so spin_until_future_complete is illegal (Kilted's executor refuses
            # reentrant spinning). Await the future instead — this yields to the
            # executor until the service response arrives.
            future = self.distance_reset_cli.call_async(req)
            await future
            if future.result() is None:
                self.get_logger().error("teach/set_dist call failed")
                result.success = False
                goal_handle.abort()
                return result

            self.mapStep = goal.map_step
            if self.mapStep <= 0.0:
                self.get_logger().warn("Record step is not positive number - changing to 1.0m")
                self.mapStep = 1.0

            try:
                os.makedirs(maps_dir(), exist_ok=True)
                map_dir = _map_path(goal.map_name)
                # Never overwrite an existing map: archive it with a timestamp
                # suffix and record a fresh one under the requested name.
                if os.path.exists(map_dir):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    backup = _map_path(f"{goal.map_name}_{ts}")
                    os.rename(map_dir, backup)
                    self.get_logger().warn(
                        "\n" + "=" * 64 +
                        f"\n  MAP '{goal.map_name}' ALREADY EXISTS — existing map renamed to"
                        f"\n  '{goal.map_name}_{ts}'; recording a fresh '{goal.map_name}'."
                        "\n" + "=" * 64)
                os.mkdir(map_dir)
                with open(_map_path(goal.map_name, "params"), "w") as f:
                    f.write(f"stepSize: {self.mapStep}\n")
                    f.write(f"cmdVelTopic: {self.cmd_vel_topic}\n")
                    f.write(f"odomTopic: {self.odom_record_topic}\n")
                    if self.gps_record_topic:
                        f.write(f"gpsTopic: {self.gps_record_topic}\n")
                    f.write(f"cameraTopic: {self._active_camera_topic}\n")
            except Exception as e:
                self.get_logger().warn(f"Unable to create map directory, ignoring: {e}")
                result.success = False
                goal_handle.abort()
                return result

            self.get_logger().info("Starting mapping")
            if not self._bag_open_for_map(goal.map_name):
                result.success = False
                goal_handle.abort()
                return result

            # Persist the start pose immediately so the bag's first
            # /recorded_odometry sample reflects the true beginning of the
            # trajectory rather than the first sample after motion crossed
            # action_dist_step.  Best-effort: if no odometry topic was
            # configured (or no message has arrived yet), skip silently —
            # this must not abort goal acceptance.
            if self.lastOdom is not None:
                now_ns = self.get_clock().now().nanoseconds
                if not self._bag_write("/recorded_odometry", self.lastOdom, now_ns):
                    self._bag_error_banner(
                        "START ODOM WRITE FAILED — bag may lack initial /recorded_odometry",
                        f"map={goal.map_name!r} odom_topic={self.odom_record_topic}",
                    )
            else:
                self.get_logger().info(
                    "No odometry sample cached at mapping start — "
                    "/recorded_odometry will start at the first motion step"
                )

            if self.lastGpsOdom is not None:
                now_ns = self.get_clock().now().nanoseconds
                if not self._bag_write("/recorded_gps", self.lastGpsOdom, now_ns):
                    self._bag_error_banner(
                        "START GPS WRITE FAILED — bag may lack initial /recorded_gps",
                        f"map={goal.map_name!r} gps_topic={self.gps_record_topic}",
                    )

            self.mapName = goal.map_name
            self._reset_record_debug_stats()
            self._apply_teach_dist_reset(0.0)
            self.isMapping = True
            self._open_repr_trace(goal.map_name)
            self._log_teach_topic_banner("teach_START")
            self._start_teach_feed_watchdog()
            pubs = 0
            try:
                pubs = self.joy_sub.get_publisher_count()
            except Exception:
                pass
            self._record_debug_log(
                f'START map={goal.map_name!r} backward={self._backward_record} '
                f'cmd_vel={self.joy_topic} pubs={pubs} '
                f'odom={self.odom_record_topic or "(none)"} '
                f'action_step={self.action_dist_step:.3f}m kp_step={self.mapStep:.3f}m'
            )
            result.success = True
            goal_handle.succeed()
            return result

        else:
            # stop mapping
            if self.target_distances is None:
                if self.header is not None and self.img_msg is not None and self.img_features is not None:
                    self._save_queue.put((
                        self.img_features, self.img_msg, self.header, self.mapName,
                        float(self.dist), self.curr_hist, self.curr_alignment,
                        self.source_map, self.save_imgs, self.bridge
                    ))
                    self._note_kp_saved(float(self.dist))
                    self.get_logger().info(f"Creating final wp at dist: {self.dist}")
                else:
                    self.get_logger().warn(
                        "Skipping final waypoint save: no synchronized image/features/header received yet")

            self.get_logger().warn(
                f"STOP STATE | header={self.header is not None} | img={self.img_msg is not None} | feat={self.img_features is not None} | dist={self.dist}"
            )
            self._dump_record_summary('STOP')
            self._stop_teach_feed_watchdog()
            self._close_repr_trace()
            self.get_logger().warn("Stopping Mapping")
            self.get_logger().info(f"Map saved under: '{os.path.abspath(_map_path(self.mapName))}'")

            time.sleep(2)
            self.isMapping = False
            # Block until every queued waypoint is flushed to disk before
            # we return success, so the map directory is complete.
            self._save_queue.join()

            self._bag_close()

            if self.target_distances is not None:
                self.get_logger().warn("Removing and copying action commands")
                dst_bag_dir = _map_path(goal.map_name, "bag")
                src_bag_dir = _map_path(goal.source_map, "bag")
                try:
                    if os.path.isdir(dst_bag_dir):
                        shutil.rmtree(dst_bag_dir)
                    shutil.copytree(src_bag_dir, dst_bag_dir)
                except Exception as e:
                    self.get_logger().warn(f"Failed to copy bag2 from source map: {e}")

            # Reverse the map if it was recorded with the backward-facing
            # camera. Bag and .npy/.jpg files must be closed first.
            if self._backward_record and self._active_map_dir:
                try:
                    self._postprocess_reverse_map(_map_path(self._active_map_dir))
                except Exception as e:
                    self.get_logger().error(f"Reverse post-processing failed: {e}")
            self._backward_record = False

            # Restore the default camera for the next session (repeat or
            # a subsequent forward teach).
            if self._active_camera_topic != self._default_camera_topic:
                await self._rebind_camera(self._default_camera_topic)

            result.success = True
            goal_handle.succeed()
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
