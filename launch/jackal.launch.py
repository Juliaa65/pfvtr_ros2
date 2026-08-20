# jackal.launch.py  --  single forward teach+repeat pipeline for this Jackal
#
# Concrete topic names for this machine: front RealSense camera1 for vision,
# Clearpath j100_0000 namespace for cmd_vel/odometry. Modeled on
# robot_template.launch.py with camera_back_topic left empty (no rear camera
# recording here — see jackal_backwards.launch.py for that).
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():

    camera_topic = DeclareLaunchArgument(
        "camera_topic",
        default_value="/camera1/camera1/color/image_raw",
        description="Front RealSense camera topic (sensor_msgs/Image)",
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
        default_value="/j100_0000/joy_teleop/cmd_vel",
        description="Topic where joystick teleop publishes commands (for recording)",
    )

    cmd_vel_robot_input = DeclareLaunchArgument(
        "cmd_vel_robot_input",
        default_value="/j100_0000/cmd_vel",
        description="Topic to send velocity commands to control the robot (twist_mux input)",
    )

    odom_topic = DeclareLaunchArgument(
        "odom_topic",
        default_value="/j100_0000/platform/odom/filtered",
        description="Topic for odometry input",
    )

    odom_record_topic = DeclareLaunchArgument(
        "odom_record_topic",
        default_value="/j100_0000/platform/odom/filtered",
        description="Topic for odometry recording in mapmaker",
    )

    particle_num = DeclareLaunchArgument("particle_num", default_value="600")
    odom_error = DeclareLaunchArgument("odom_error", default_value="0.025")
    dist_init_std = DeclareLaunchArgument("dist_init_std", default_value="1.0")
    align_beta = DeclareLaunchArgument("align_beta", default_value="10.0")
    align_init_std = DeclareLaunchArgument("align_init_std", default_value="1.0")
    add_random = DeclareLaunchArgument("add_random", default_value="0.05")
    model_path = DeclareLaunchArgument("model_path", default_value="")

    choice_beta = DeclareLaunchArgument("choice_beta", default_value="2.3")
    position_estimator = DeclareLaunchArgument(
        "position_estimator",
        default_value="kde",
        description=(
            "PF2D output estimator. 'kde' picks the dominant mode (default, "
            "correct under multimodal posteriors). 'weighted_mean' is the legacy "
            "centroid (drifts between modes but well-tested)."
        ),
    )
    kde_grid_res = DeclareLaunchArgument("kde_grid_res", default_value="64")
    kde_align_span = DeclareLaunchArgument(
        "kde_align_span",
        default_value="1.0",
        description="±m around published d for alignment KDE; <=0 disables (all particles).",
    )
    kde_min_align_frac = DeclareLaunchArgument("kde_min_align_frac", default_value="0.08")
    kde_max_step_back = DeclareLaunchArgument("kde_max_step_back", default_value="0.1")
    kde_max_step_fwd = DeclareLaunchArgument("kde_max_step_fwd", default_value="0.1")
    final_window_dist = DeclareLaunchArgument(
        "final_window_dist",
        default_value="1.5",
        description=(
            "Distance to map end (m) at which visual distance weighting, "
            "resampling and random respawn are gated off so odometry can "
            "carry the estimate to the finish. Latches until the next repeat."
        ),
    )
    matching_type = DeclareLaunchArgument("matching_type", default_value="siam")
    turn_gain = DeclareLaunchArgument("turn_gain", default_value="1.0")

    navigation_method = DeclareLaunchArgument(
        "navigation_method",
        default_value="pf2d",
        description=(
            "Repeat-phase fusion method. "
            "'classic' = Bearnav Classic image-based correction "
            "(requires GUI/action image_pub == 0). "
            "'pf2d' = particle filter using PF2D parameters "
            "(particle_num, odom_error, etc.; requires image_pub >= 1)."
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
                    "kde_max_step_back": lc("kde_max_step_back"),
                    "kde_max_step_fwd": lc("kde_max_step_fwd"),
                    "final_window_dist": lc("final_window_dist"),
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
        kde_max_step_back,
        kde_max_step_fwd,
        final_window_dist,
        matching_type,
        turn_gain,
        navigation_method,
        pfvtr_group,
    ])
