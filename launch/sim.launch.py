from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():

    camera_topic = DeclareLaunchArgument(
        "camera_topic",
        default_value="/camera_front_publisher",
        description="Camera topic name",
    )

    camera_back_topic = DeclareLaunchArgument(
        "camera_back_topic",
        default_value="",
        description=(
            "Rear-facing camera topic used when the MapMaker action is "
            "called with record_backward=true. Empty (the default) disables "
            "backward mapping — such goals will be rejected with an error."
        ),
    )

    cmd_vel_teleop_output = DeclareLaunchArgument(
        "cmd_vel_teleop_output",
        default_value="/cmd_vel_publisher",
        description="Topic where simulator publishes teleop commands (for recording)",
    )

    cmd_vel_robot_input = DeclareLaunchArgument(
        "cmd_vel_robot_input",
        default_value="/cmd_vel_subscriber",
        description="Topic to send velocity commands to control the robot",
    )

    odom_topic = DeclareLaunchArgument(
        "odom_topic",
        default_value="/odometry_publisher",
        description="Topic for odometry input",
    )

    odom_record_topic = DeclareLaunchArgument(
        "odom_record_topic",
        default_value="/odometry_publisher",
        description="Topic for odometry recording in mapmaker",
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
    turn_gain = DeclareLaunchArgument("turn_gain", default_value="0.05")


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
                executable="controller-ros-2.py",
                name="controller",
                output="screen",
                respawn=True,
                parameters=[{
                    "cmd_vel_topic": lc("cmd_vel_robot_input"),
                    "turn_gain": lc("turn_gain"),
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
                    "camera_back_topic": lc("camera_back_topic"),
                    "cmd_vel_topic": lc("cmd_vel_teleop_output"),
                    "odom_record_topic": lc("odom_record_topic"),
                }]
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
        ]
    )

    return LaunchDescription([
        camera_topic,
        camera_back_topic,
        cmd_vel_teleop_output,
        cmd_vel_robot_input,
        odom_topic,
        odom_record_topic,
        particle_num,
        odom_error,
        dist_init_std,
        align_beta,
        align_init_std,
        add_random,
        model_path,
        choice_beta,
        matching_type,
        turn_gain,
        pfvtr_group,
    ])
