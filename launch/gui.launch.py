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
    # `odom_topic` and `camera_topic` ARE launch args, because they are
    # connection endpoints: the GUI's rate-readiness measurement subscribes
    # to both, and the topic names must match the robot's launch args of
    # the same name. Keeping them in launch files (rather than fetching
    # them at runtime) makes the wiring readable and removes async startup
    # complexity.

    odom_topic = DeclareLaunchArgument(
        "odom_topic",
        default_value="/robot1/odometry",
        description=(
            "Odometry topic the GUI subscribes to for the rate-readiness "
            "measurement. Must match the `odom_topic` passed to the robot's "
            "navigation-stack launch."
        ),
    )

    camera_topic = DeclareLaunchArgument(
        "camera_topic",
        default_value="/robot1/camera1/image",
        description=(
            "Camera topic the GUI subscribes to for the rate-readiness "
            "measurement. Must match the `camera_topic` passed to the robot's "
            "navigation-stack launch."
        ),
    )

    lc = LaunchConfiguration

    return LaunchDescription([
        odom_topic,
        camera_topic,
        Node(
            package="pfvtr",
            executable="vtr_gui.py",
            name="vtr_gui",
            output="screen",
            respawn=True,
            parameters=[{
                "odom_topic": lc("odom_topic"),
                "camera_topic": lc("camera_topic"),
            }],
        ),
    ])
