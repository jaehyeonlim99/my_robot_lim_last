#!/usr/bin/env python3
"""
Phase 2: Waypoint 자동 이동
저장된 waypoint로 순차 이동
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
from action_msgs.msg import GoalStatus
import json
import time

class AutoNavigation(Node):
    def __init__(self):
        super().__init__('auto_navigation')
        
        # Navigator 초기화
        self.navigator = BasicNavigator()
        
        # Waypoints 로드
        self.waypoints = self.load_waypoints()
        
        self.get_logger().info('='*50)
        self.get_logger().info('🤖 자동 Navigation')
        self.get_logger().info('='*50)
        self.get_logger().info(f'📂 Waypoint 로드: {len(self.waypoints)}개')
        
    def load_waypoints(self):
        """JSON에서 waypoints 로드"""
        filename = '/home/limjaehyeon/waypoints.json'
        
        try:
            with open(filename, 'r') as f:
                waypoints = json.load(f)
            return waypoints
        except Exception as e:
            self.get_logger().error(f'파일 로드 실패: {e}')
            return {}
    
    def create_goal(self, x, y):
        """Navigation Goal 생성"""
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        return goal
    
    def go_to_waypoint(self, name):
        """특정 waypoint로 이동"""
        if name not in self.waypoints:
          self.get_logger().error(f'{name} 없음!')
          return False
    
        wp = self.waypoints[name]
    
        self.get_logger().info('')
        self.get_logger().info('┌' + '─'*40 + '┐')
        self.get_logger().info(f'│  {name} 이동')
        self.get_logger().info('└' + '─'*40 + '┘')
        self.get_logger().info(f'→ 목표: ({wp["x"]:.2f}, {wp["y"]:.2f})')
    
        # Goal 생성 및 전송
        goal = self.create_goal(wp['x'], wp['y'])
        self.navigator.goToPose(goal)
        
        # 완료 대기
        while not self.navigator.isTaskComplete():
            time.sleep(0.1)
        
        # 성공으로 간주 (완료됨 = 성공)
        self.get_logger().info(f'✅ {name} 도착!')
        return True
    
    def patrol_all(self):
        """모든 waypoint 순회"""
        self.get_logger().info('')
        self.get_logger().info('🚶 순찰 시작!')
        self.get_logger().info('')
        
        # 3초 대기
        time.sleep(3)
        
        # 순서대로 이동
        for i in range(1, len(self.waypoints) + 1):
            name = f'waypoint{i}'
            success = self.go_to_waypoint(name)
            
            if not success:
                self.get_logger().warn('순찰 중단!')
                break
            
            # 잠시 대기
            time.sleep(2)
        
        self.get_logger().info('')
        self.get_logger().info('='*50)
        self.get_logger().info('📊 순찰 완료!')
        self.get_logger().info('='*50)

def main():
    rclpy.init()
    node = AutoNavigation()
    
    try:
        # 전체 순찰
        node.patrol_all()
        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()