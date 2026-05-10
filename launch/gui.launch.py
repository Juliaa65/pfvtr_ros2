from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Standalone launch for the VT&R GUI.
    #
    # Use case: GUI runs on the operator's laptop while the navigation stack
    # runs on the robot, joined via Zenoh (rmw_zenoh_cpp + a router on the
    # robot, ZENOH_ROUTER_CHECK_ATTEMPTS / session config exporting the
    # router address on the laptop). The Zenoh setup, RMW_IMPLEMENTATION,
    # and ROS_DOMAIN_ID belong in the user's shell — not in this launch
    # file — because they vary per deployment.
    #
    # No `PushRosNamespace`: the laptop GUI talks *to* the robot's `pfvtr`
    # namespace via absolute service/topic paths; it isn't logically part
    # of that namespace.
    #
    # `navigation_method` is intentionally NOT a launch arg — the GUI fetches
    # it from /pfvtr/sensors at startup so the robot is the single source of
    # truth for that semantic config.
    #
    # `odom_topic` and `camera_info_topic` ARE launch args, because they are
    # connection endpoints: the GUI's rate-readiness measurement subscribes
    # to both, and the topic names must match the robot's launch args of
    # the same name. Keeping them in launch files (rather than fetching
    # them at runtime) makes the wiring readable and removes async startup
    # complexity.

    odom_topic = DeclareLaunchArgument(
        "odom_topic",
        default_value="",
        description=(
            "Odometry topic the GUI subscribes to for the rate-readiness "
            "measurement. Must match the `odom_topic` passed to the robot's "
            "navigation-stack launch."
        ),
    )

    camera_info_topic = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="",
        description=(
            "CameraInfo topic the GUI subscribes to for the rate-readiness "
            "measurement. Must match the `camera_info_topic` passed to the "
            "robot's navigation-stack launch."
        ),
    )

    # Symmetric ± spans for the GUI's "Displace" checkbox. Adds a random
    # offset drawn from [-span, +span] to the map's recorded start pose
    # when the user clicks Teleport with Displace checked.
    displace_x_span = DeclareLaunchArgument("displace_x_span", default_value="2.0")
    displace_y_span = DeclareLaunchArgument("displace_y_span", default_value="2.0")
    displace_yaw_span_deg = DeclareLaunchArgument("displace_yaw_span_deg", default_value="20.0")

    lc = LaunchConfiguration

    return LaunchDescription([
        odom_topic,
        camera_info_topic,
        displace_x_span,
        displace_y_span,
        displace_yaw_span_deg,
        Node(
            package="pfvtr",
            executable="vtr_gui.py",
            name="vtr_gui",
            output="screen",
            respawn=True,
            parameters=[{
                "odom_topic": lc("odom_topic"),
                "camera_info_topic": lc("camera_info_topic"),
                "displace_x_span": lc("displace_x_span"),
                "displace_y_span": lc("displace_y_span"),
                "displace_yaw_span_deg": lc("displace_yaw_span_deg"),
            }],
        ),
    ])
