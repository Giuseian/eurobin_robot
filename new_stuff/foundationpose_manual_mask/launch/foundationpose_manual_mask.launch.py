#!/usr/bin/env python3
# Launch minimale per Isaac ROS FoundationPose: contiene SOLO foundationpose_node,
# niente catena RT-DETR/resize/pad. Usalo insieme a photo_mask_publisher.py,
# che pubblica direttamente sui topic qui sotto rimappati (rgb/image_rect_color,
# depth_image, rgb/camera_info, segmentation).

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    mesh_file_path = LaunchConfiguration('mesh_file_path')
    texture_path = LaunchConfiguration('texture_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')

    foundationpose_node = ComposableNode(
        name='foundationpose_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'texture_path': texture_path,

            'refine_engine_file_path': refine_engine_file_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],

            'score_engine_file_path': score_engine_file_path,
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names': ['input1', 'input2'],
            'score_output_tensor_names': ['output_tensor'],
            'score_output_binding_names': ['output1'],
        }],
        remappings=[
            ('pose_estimation/depth_image', 'depth_image'),
            ('pose_estimation/image', 'rgb/image_rect_color'),
            ('pose_estimation/camera_info', 'rgb/camera_info'),
            ('pose_estimation/segmentation', 'segmentation'),
            ('pose_estimation/output', 'output'),
        ],
    )

    container = ComposableNodeContainer(
        name='foundationpose_manual_mask_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[foundationpose_node],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('mesh_file_path', default_value=''),
        DeclareLaunchArgument('texture_path', default_value=''),
        DeclareLaunchArgument('refine_engine_file_path', default_value=''),
        DeclareLaunchArgument('score_engine_file_path', default_value=''),
        container,
    ])
