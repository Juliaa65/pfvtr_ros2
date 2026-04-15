cmake_minimum_required(VERSION 3.10)
project(pfvtr)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(action_msgs REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/ImageList.msg"
  "msg/FloatList.msg"
  "msg/FeatureLocations.msg"
  "msg/SensorsInput.msg"
  "msg/SensorsOutput.msg"
  "msg/DistancedTwist.msg"
  "msg/Features.msg"
  "msg/SensorsInputImages.msg"
  "msg/FeaturesList.msg"
  "msg/Descriptor.msg"
  "msg/Histogram.msg"
  "msg/HistogramList.msg"

  # Services
  "srv/Alignment.srv"
  "srv/SetDist.srv"
  "srv/SetClockGain.srv"
  "srv/Representations.srv"
  "srv/StopRepeater.srv"

  # Actions
  "action/MapMaker.action"
  "action/MapRepeater.action"

  DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    builtin_interfaces
    action_msgs
)


install(PROGRAMS
  src/controller/controller-ros-2.py
  src/master/repeater-ros-2.py
  src/master/mapmaker-ros-2.py
  src/master/vtr_gui.py
  src/sensors/sensors-ros-2.py
  src/sensors/representations-ros-2.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

endif()

ament_package()
