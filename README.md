# 고인돌 유적지 자율 순찰 로봇

전북 피지컬 AI 기반 고창 고인돌 유적지 무인 순찰 시스템

## 🎯 프로젝트 개요

- **목적**: 고인돌 유적지 24시간 무인 순찰
- **기술**: ROS2 Humble, Gazebo, Navigation2, SLAM, YOLOv8
- **로봇**: TurtleBot3 Waffle (2배 확대)

## 🚀 주요 기능

### Phase 1: 수동 탐색
- 키보드로 로봇 조종
- 's' 키로 waypoint 저장
- JSON 파일 생성

### Phase 2: 자동 순찰
- 저장된 waypoint 자동 순회
- Navigation2 경로 계획
- 장애물 회피

## 🛠️ 설치 및 실행

### 빌드
```bash
cd ~/my_robot_lim
colcon build --packages-select my_robot
source install/setup.bash
```

### 1. Gazebo 실행
```bash
ros2 launch my_robot gazebo_with_robot.launch.py
```

### 2. SLAM 실행
```bash
ros2 launch slam_toolbox online_async_launch.py
```

### 3. Phase 1: 수동 탐색
```bash
python3 ~/my_robot_lim/src/my_robot/scripts/manual_exploration.py
```

### 4. Phase 2: 자동 순찰
```bash
python3 ~/my_robot_lim/src/my_robot/scripts/auto_navigation.py
```

## 📊 결과

### Waypoint 저장 예시
```json
{
  "waypoint1": {"x": -4.86, "y": 5.21, "z": 0.0},
  "waypoint2": {"x": 6.20, "y": 4.04, "z": 0.0},
  "waypoint3": {"x": 7.15, "y": -4.87, "z": 0.0}
}
```

## 👥 팀원

- 효선님: 맵 제작, 월드 파일
- 재현님: ROS2 통합, Navigation

## 📄 라이센스

MIT License
