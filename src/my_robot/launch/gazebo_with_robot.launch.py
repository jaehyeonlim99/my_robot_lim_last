from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    # URDF 파일 경로
    urdf_file = os.path.join(
        os.path.dirname(__file__), '..', 'urdf', 'turtlebot3_with_sensors.urdf'
    )
    
    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    # World 파일 경로 인자
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='/home/limjaehyeon/save/wall/park_world.world',  # ✅ park_world
        description='Full path to world file to load'
    )

    # Gazebo 서버 시작 (커스텀 월드 사용)
    gzserver = ExecuteProcess(
        cmd=['gzserver', '--verbose',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             LaunchConfiguration('world')],
        output='screen'
    )

    # Gazebo 클라이언트 시작
    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen'
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }]
    )

    # Spawn Robot (Gazebo 시작 10초 후)
    spawn_entity_delayed = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', 'my_robot',
                    '-file', urdf_file,
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '2.0'
                ],
                output='screen'
            )
        ]
    )

    # PointCloud to LaserScan
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        parameters=[{
            'target_frame': 'base_scan',
            'transform_tolerance': 0.01,
            'min_height': 0.15,
            'max_height': 1.8,
            'angle_min': -3.14159,
            'angle_max': 3.14159,
            'angle_increment': 0.0087,
            'scan_time': 0.1,
            'range_min': 0.3,
            'range_max': 30.0,
            'use_inf': True,
            'inf_epsilon': 1.0
        }],
        remappings=[
            ('cloud_in', '/velodyne_points'),
            ('scan', '/scan_filtered')
        ],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn_entity_delayed,  # 10초 딜레이
        pointcloud_to_laserscan
    ])