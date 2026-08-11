"""Dual-namespace pfvtr template — front teach+repeat plus rear-camera map recording.

Copy this file per-robot (like robot_template.launch.py) and fill in the
REQUIRED topic args. It starts TWO independent pfvtr pipelines from a single
launch:

  /pfvtr      - full teach+repeat stack (sensors, representations, mapmaker,
                repeater, controller), used for the normal forward map.
  /pfvtr_map  - mapping-only stack (sensors, representations, mapmaker), used
                to record a second map on the rear camera.

A single `representations` node can only bind one camera at a time, so
recording the rear camera while `/pfvtr` maps/repeats the front camera needs
its own sensors+representations+mapmaker pipeline. Both pipelines share the
same topic/tuning args below; `/pfvtr_map`'s mapmaker rebinds itself to
`camera_back_topic` internally when a MapMaker goal arrives with
`record_backward: true` (goal is rejected if camera_back_topic is empty).
On save, that map is post-processed (waypoints reversed, turns negated) so it
can later be repeated forward, after a physical 180-degree turn, to retrace
the path backward.

Trigger recording on the second pipeline with, e.g.:
  ros2 action send_goal /pfvtr_map/mapmaker pfvtr/action/MapMaker \
    "{map_name: 'map_to_a', start: true, map_step: 1.0, record_backward: true}"

Both sessions start together and stop together with this one launch process.
If you need to stop the rear pipeline independently to save compute (its
representations node extracts features every frame, so it's only needed
during the teach pass), run the two pipelines as separate `ros2 launch`
invocations instead — see bringup/launch/pfvtr.launch.py and
bringup/launch/pfvtr_map.launch.py for that pattern.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():

    camera_topic = DeclareLaunchArgument(
        "camera_topic",
        default_value="REQUIRED",
        description="Front camera topic name",
    )

    camera_back_topic = DeclareLaunchArgument(
        "camera_back_topic",
        default_value="REQUIRED",
        description=(
            "Rear-facing camera topic, rebound in by the /pfvtr_map mapmaker "
            "when its MapMaker action is called with record_backward=true. "
            "This launch file exists to record that backward map, so this "
            "must be set."
        ),
    )

    cmd_vel_teleop_output = DeclareLaunchArgument(
        "cmd_vel_teleop_output",
        default_value="REQUIRED",
        description="Topic where teleop/joystick publishes commands (for recording)",
    )

    cmd_vel_robot_input = DeclareLaunchArgument(
        "cmd_vel_robot_input",
        default_value="REQUIRED",
        description="Topic to send velocity commands to control the robot",
    )

    odom_topic = DeclareLaunchArgument(
        "odom_topic",
        default_value="REQUIRED",
        description="Topic for odometry input",
    )

    odom_record_topic = DeclareLaunchArgument(
        "odom_record_topic",
        default_value="",
        description="Topic for odometry recording in mapmaker",
    )

    particle_num = DeclareLaunchArgument("particle_num", default_value="600")
    odom_error = DeclareLaunchArgument("odom_error", default_value="0.025")
    dist_init_std = DeclareLaunchArgument("dist_init_std", default_value="1.0")
    align_beta = DeclareLaunchArgument("align_beta", default_value="10.0")
    align_init_std = DeclareLaunchArgument("align_init_std", default_value="1.0")
    add_random = DeclareLaunchArgument("add_random", default_value="0.05")
    model_path = DeclareLaunchArgument("model_path", default_value="")

    choice_beta = DeclareLaunchArgument("choice_beta", default_value="2.5")
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
    matching_type = DeclareLaunchArgument("matching_type", default_value="siam")
    turn_gain = DeclareLaunchArgument("turn_gain", default_value="1.0")

    navigation_method = DeclareLaunchArgument(
        "navigation_method",
        default_value="classic",
        description=(
            "Repeat-phase fusion method. "
            "'classic' = Bearnav Classic image-based correction "
            "(requires GUI/action image_pub == 0). "
            "'pf2d' = particle filter using PF2D parameters "
            "(particle_num, odom_error, etc.; requires image_pub >= 1)."
        ),
    )

    lc = LaunchConfiguration

    sensors_params = [{
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
        "matching_type": lc("matching_type"),
        "model_path": lc("model_path"),
        "navigation_method": lc("navigation_method"),
    }]

    representations_params = [{
        "camera_topic": lc("camera_topic"),
        "matching_type": lc("matching_type"),
        "model_path": lc("model_path"),
    }]

    mapmaker_params = [{
        "camera_topic": lc("camera_topic"),
        "camera_back_topic": lc("camera_back_topic"),
        "cmd_vel_topic": lc("cmd_vel_teleop_output"),
        "odom_record_topic": lc("odom_record_topic"),
    }]

    pfvtr_group = GroupAction(
        [
            PushRosNamespace("pfvtr"),

            Node(
                package="pfvtr",
                executable="sensors-ros-2.py",
                name="sensors",
                output="screen",
                respawn=True,
                parameters=sensors_params,
            ),

            Node(
                package="pfvtr",
                executable="representations-ros-2.py",
                name="representations",
                output="screen",
                respawn=True,
                parameters=representations_params,
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
                parameters=mapmaker_params,
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

    pfvtr_map_group = GroupAction(
        [
            PushRosNamespace("pfvtr_map"),

            Node(
                package="pfvtr",
                executable="sensors-ros-2.py",
                name="sensors",
                output="screen",
                respawn=True,
                parameters=sensors_params,
            ),

            Node(
                package="pfvtr",
                executable="representations-ros-2.py",
                name="representations",
                output="screen",
                respawn=True,
                parameters=representations_params,
            ),

            Node(
                package="pfvtr",
                executable="mapmaker-ros-2.py",
                name="mapmaker",
                output="screen",
                respawn=True,
                parameters=mapmaker_params,
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
        matching_type,
        turn_gain,
        navigation_method,
        pfvtr_group,
        pfvtr_map_group,
    ])
