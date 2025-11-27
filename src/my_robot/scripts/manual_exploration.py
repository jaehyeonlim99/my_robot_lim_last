#!/usr/bin/env python3
"""
Phase 1: 수동 탐색
키보드로 로봇 조종하며 waypoint 저장
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import json
import sys
import select
import tty
import termios

class ManualExploration(Node):
    def __init__(self):
        super().__init__('manual_exploration')
        
        # Odometry 구독
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.current_pose = None
        self.waypoints = {}
        self.waypoint_count = 0
        
        self.get_logger().info('='*50)
        self.get_logger().info('🎮 수동 탐색 모드')
        self.get_logger().info('='*50)
        self.get_logger().info('')
        self.get_logger().info('명령어:')
        self.get_logger().info("  's' → 현재 위치 저장")
        self.get_logger().info("  'l' → 저장 목록")
        self.get_logger().info("  'd' → 완료 (JSON 저장)")
        self.get_logger().info("  'q' → 종료")
        self.get_logger().info('')
        self.get_logger().info('🚀 준비 완료! 키보드로 탐색하세요.')
        
        # 키보드 입력용
        self.settings = termios.tcgetattr(sys.stdin)
        
    def odom_callback(self, msg):
        """Odometry 콜백 - 현재 위치 저장"""
        self.current_pose = msg.pose.pose
        
    def save_waypoint(self):
        """현재 위치를 waypoint로 저장"""
        if self.current_pose is None:
            self.get_logger().warn('위치 정보 없음!')
            return
        
        self.waypoint_count += 1
        name = f'waypoint{self.waypoint_count}'
        
        self.waypoints[name] = {
            'x': self.current_pose.position.x,
            'y': self.current_pose.position.y,
            'z': self.current_pose.position.z
        }
        
        self.get_logger().info(f'✅ {name} 저장!')
        self.get_logger().info(f'   위치: ({self.current_pose.position.x:.2f}, '
                              f'{self.current_pose.position.y:.2f})')
    
    def list_waypoints(self):
        """저장된 waypoint 목록"""
        self.get_logger().info('')
        self.get_logger().info('📍 저장된 Waypoint: ' + str(len(self.waypoints)) + '개')
        for name, pos in self.waypoints.items():
            self.get_logger().info(f'  {name}: ({pos["x"]:.2f}, {pos["y"]:.2f})')
        self.get_logger().info('')
    
    def save_to_file(self):
        """JSON 파일로 저장"""
        filename = '/home/limjaehyeon/waypoints.json'
        
        with open(filename, 'w') as f:
            json.dump(self.waypoints, f, indent=2)
        
        self.get_logger().info('')
        self.get_logger().info('💾 저장 완료!')
        self.get_logger().info(f'   파일: {filename}')
        self.get_logger().info('')
    
    def get_key(self):
        """키 입력 받기"""
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

def main():
    rclpy.init()
    node = ManualExploration()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # 키 입력 확인
            if select.select([sys.stdin], [], [], 0)[0]:
                key = node.get_key()
                
                if key == 's':
                    node.save_waypoint()
                elif key == 'l':
                    node.list_waypoints()
                elif key == 'd':
                    node.save_to_file()
                    node.list_waypoints()
                elif key == 'q':
                    node.get_logger().info('종료!')
                    break
                    
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()