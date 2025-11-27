#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class SimpleDetector(Node):
    def __init__(self):
        super().__init__('simple_detector')
        self.get_logger().info('Loading YOLO...')
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image,
            '/depth_camera/depth_camera/image_raw',
            self.callback,
            10
        )
        self.get_logger().info('Detector ready!')
        self.frame_count = 0
    
    def callback(self, msg):
        try:
            self.frame_count += 1
            
            # 10프레임마다 로그
            if self.frame_count % 10 == 0:
                self.get_logger().info(f'Processing frame {self.frame_count}...')
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            results = self.model(cv_image, verbose=False)
            
            # 감지된 객체 수
            num_detections = len(results[0].boxes)
            
            if num_detections > 0:
                self.get_logger().info(f'Found {num_detections} objects!')
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    name = self.model.names[cls]
                    conf = float(box.conf[0])
                    
                    # confidence 관계없이 모두 출력
                    self.get_logger().info(f'{name}: {conf:.2f}')
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main():
    rclpy.init()
    node = SimpleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()