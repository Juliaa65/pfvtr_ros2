from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # sim_gui.launch.py + the topic_diagnostics sanity monitor.
    #
    # This is the ONLY launch file that starts the diagnostics node. The base
    # stack is brought up by including sim_gui.launch.py unchanged, so there is
    # no duplication of node/parameter wiring — only the extra diagnostics node
    # lives here.
    #
    # The three topics the diagnostics node watches are also sim_gui launch
    # args; they are re-declared here (same defaults) and forwarded to the
    # include so overriding e.g. camera_topic stays consistent across the
    # stack and the monitor.

    camera_topic = DeclareLaunchArgument(
        "camera_topic",
        default_value="/camera_front_publisher",
        description="Image topic — also forwarded to the included sim_gui stack.",
    )
    odom_topic = DeclareLaunchArgument(
        "odom_topic",
        default_value="/odometry_publisher",
        description="Odometry topic — also forwarded to the included sim_gui stack.",
    )
    cmd_vel_teleop_output = DeclareLaunchArgument(
        "cmd_vel_teleop_output",
        default_value="/cmd_vel_subscriber",
        description="Teleop cmd_vel topic — also forwarded to the included sim_gui stack.",
    )

    lc = LaunchConfiguration

    sim_gui_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("pfvtr"), "launch", "sim_gui.launch.py"]
            )
        ),
        launch_arguments={
            "camera_topic": lc("camera_topic"),
            "odom_topic": lc("odom_topic"),
            "cmd_vel_teleop_output": lc("cmd_vel_teleop_output"),
        }.items(),
    )

    diagnostics_group = GroupAction(
        [
            PushRosNamespace("pfvtr"),
            Node(
                package="pfvtr",
                executable="topic_diagnostics.py",
                name="sanity_monitor",
                output="screen",
                parameters=[{
                    "camera_topic": lc("camera_topic"),
                    "odom_topic": lc("odom_topic"),
                    "cmd_vel_sub_topic": lc("cmd_vel_teleop_output"),
                }],
            ),
        ]
    )

    return LaunchDescription([
        camera_topic,
        odom_topic,
        cmd_vel_teleop_output,
        sim_gui_launch,
        diagnostics_group,
    ])
