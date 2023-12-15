from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    config_path = "/home/tahnt/T3_Repos/post_process_packages/ros2_ws/src/lane_detection_package/config/sliding_box_ld.yaml"
    
    return LaunchDescription([
        Node(
            package="lane_detection_package",
            executable="sliding_box_ld",
            name="sliding_box_lane_detection",
            output="screen",
            emulate_tty=True,
            parameters=[config_path]
        ),
    ])