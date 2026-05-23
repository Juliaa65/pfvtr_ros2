from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():

    camera_topic = DeclareLaunchArgument(
        "camera_topic",
        default_value="/camera_front_publisher",
        description="Image topic consumed by representations, mapmaker, and repeater.",
    )

    camera_back_topic = DeclareLaunchArgument(
        "camera_back_topic",
        default_value="",
        description=(
            "Rear-facing camera topic used when the MapMaker action is "
            "called with record_backward=true. Empty disables backward mapping."
        ),
    )

    cmd_vel_teleop_output = DeclareLaunchArgument(
        "cmd_vel_teleop_output",
        default_value="/cmd_vel_subscriber",
        description="Topic where simulator publishes teleop commands for map recording.",
    )

    cmd_vel_robot_input = DeclareLaunchArgument(
        "cmd_vel_robot_input",
        default_value="/cmd_vel_subscriber",
        description="Topic where PFVTR controller publishes velocity commands.",
    )

    odom_topic = DeclareLaunchArgument(
        "odom_topic",
        default_value="/odometry_publisher",
        description="Odometry topic consumed by sensors.",
    )

    odom_record_topic = DeclareLaunchArgument(
        "odom_record_topic",
        default_value="/odometry_publisher",
        description="Odometry topic recorded by mapmaker.",
    )

    particle_num = DeclareLaunchArgument("particle_num", default_value="600")
    odom_error = DeclareLaunchArgument("odom_error", default_value="0.025")
    dist_init_std = DeclareLaunchArgument("dist_init_std", default_value="1.0")
    align_beta = DeclareLaunchArgument("align_beta", default_value="10.0")
    align_init_std = DeclareLaunchArgument("align_init_std", default_value="1.0")
    add_random = DeclareLaunchArgument("add_random", default_value="0.01")
    model_path = DeclareLaunchArgument("model_path", default_value="")

    choice_beta = DeclareLaunchArgument("choice_beta", default_value="2.5")
    position_estimator = DeclareLaunchArgument(
        "position_estimator",
        default_value="kde",
        description=(
            "PF2D output estimator. 'kde' picks the dominant mode. "
            "'weighted_mean' is the legacy centroid estimator."
        ),
    )
    kde_grid_res = DeclareLaunchArgument("kde_grid_res", default_value="64")
    kde_align_span = DeclareLaunchArgument("kde_align_span", default_value="0.5")
    kde_min_align_frac = DeclareLaunchArgument("kde_min_align_frac", default_value="0.08")
    matching_type = DeclareLaunchArgument("matching_type", default_value="siam")

    turn_gain = DeclareLaunchArgument("turn_gain", default_value="1.0")
    velocity_gain = DeclareLaunchArgument("velocity_gain", default_value="1.0")

    navigation_method = DeclareLaunchArgument(
        "navigation_method",
        default_value="pf2d",
        description=(
            "Repeat-phase fusion method. "
            "'classic' = Bearnav Classic image-based correction "
            "(requires action image_pub == 0). "
            "'pf2d' = particle-filter repeat using PF2D parameters "
            "(requires image_pub >= 1)."
        ),
    )

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
                    "position_estimator": lc("position_estimator"),
                    "kde_grid_res": lc("kde_grid_res"),
                    "kde_align_span": lc("kde_align_span"),
                    "kde_min_align_frac": lc("kde_min_align_frac"),
                    "matching_type": lc("matching_type"),
                    "model_path": lc("model_path"),
                    "navigation_method": lc("navigation_method"),
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
                    "velocity_gain": lc("velocity_gain"),
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
        position_estimator,
        kde_grid_res,
        kde_align_span,
        kde_min_align_frac,
        matching_type,
        turn_gain,
        velocity_gain,
        navigation_method,
        pfvtr_group,
    ])