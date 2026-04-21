# hellhest-jr.launch.py  --  launch for the Jr simulator with standard topic names
# This file is intentionally excluded from git (.gitignore) so it won't be
# overwritten by upstream pulls.  Edit topic defaults here freely.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():

    camera_topic = DeclareLaunchArgument(
        "camera_topic",
        default_value="/front_left/image_raw",
        description="sensor_msgs/Image topic from the simulator",
    )

    cmd_vel_pub = DeclareLaunchArgument(
        "cmd_vel_pub",
        default_value="/cmd_vel",
        description="cmd_vel output during repeat (controller publishes Twist here)",
    )

    cmd_vel_sub = DeclareLaunchArgument(
        "cmd_vel_sub",
        default_value="/cmd_vel",
        description="cmd_vel source to record while teaching a map",
    )

    odom_topic = DeclareLaunchArgument(
        "odom_topic",
        default_value="/odom",
        description="nav_msgs/Odometry for distance fusion (sensors node)",
    )

    additional_record_topics = DeclareLaunchArgument(
        "additional_record_topics",
        default_value="/odom",
        description="Odometry topic for mapmaker bag /recorded_odometry",
    )

    particle_num = DeclareLaunchArgument("particle_num", default_value="600")
    odom_error = DeclareLaunchArgument("odom_error", default_value="0.025")
    dist_init_std = DeclareLaunchArgument("dist_init_std", default_value="1.0")
    align_beta = DeclareLaunchArgument("align_beta", default_value="10.0")
    align_init_std = DeclareLaunchArgument("align_init_std", default_value="1.0")
    add_random = DeclareLaunchArgument("add_random", default_value="0.01")
    model_path = DeclareLaunchArgument("model_path", default_value="")

    choice_beta = DeclareLaunchArgument("choice_beta", default_value="2.5")
    matching_type = DeclareLaunchArgument("matching_type", default_value="siam")


    lc = LaunchConfiguration


    pfvtr_group = GroupAction(
        [
            PushRosNamespace("pfvtr"),

            Node(
                package="pfvtr",
                executable="sensors-ros-2.py",
                name="sensors",
                output="screen",
                respawn=True,
                parameters=[{
                    "odom_topic": lc("odom_topic"),
                    "particle_num": lc("particle_num"),
                    "odom_error": lc("odom_error"),
                    "dist_init_std": lc("dist_init_std"),
                    "align_beta": lc("align_beta"),
                    "align_init_std": lc("align_init_std"),
                    "choice_beta": lc("choice_beta"),
                    "add_random": lc("add_random"),
                    "matching_type": lc("matching_type"),
                    "model_path": lc("model_path"),
                }],
            ),

            Node(
                package="pfvtr",
                executable="representations-ros-2.py",
                name="representations",
                output="screen",
                respawn=True,
                parameters=[{
                    "camera_topic": lc("camera_topic"),
                    "matching_type": lc("matching_type"),
                    "model_path": lc("model_path"),
                }],
            ),

            Node(
                package="pfvtr",
                executable="mapmaker-ros-2.py",
                name="mapmaker",
                output="screen",
                respawn=True,
                parameters=[{
                    "camera_topic": lc("camera_topic"),
                    "cmd_vel_topic": lc("cmd_vel_sub"),
                    "additional_record_topics": lc("additional_record_topics"),
                }],
            ),

            Node(
                package="pfvtr",
                executable="repeater-ros-2.py",
                name="repeater",
                output="screen",
                respawn=True,
                parameters=[{
                    "camera_topic": lc("camera_topic"),
                }],
            ),

            Node(
                package="pfvtr",
                executable="controller-ros-2.py",
                name="controller",
                output="screen",
                respawn=True,
                parameters=[{
                    "cmd_vel_topic": lc("cmd_vel_pub"),
                }],
            ),
        ]
    )

    return LaunchDescription([
        camera_topic,
        cmd_vel_pub,
        cmd_vel_sub,
        odom_topic,
        additional_record_topics,
        particle_num,
        odom_error,
        dist_init_std,
        align_beta,
        align_init_std,
        add_random,
        model_path,
        choice_beta,
        matching_type,
        pfvtr_group,
    ])
