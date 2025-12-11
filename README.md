# 고창 고인돌 유적지 자율 순찰 로봇 (ROS2 시뮬레이터)

ROS2 기반 **고창 고인돌 유적지 무인 순찰 로봇 시뮬레이션 프로젝트**입니다.  
Gazebo + Navigation2 + SLAM + LiDAR/RGB 센서, 그리고 VLM(시각-언어 모델) 연동을 위한 기반 코드를 포함한 워크스페이스입니다.

---

## 1. 프로젝트 개요

### 🎯 목적

문화재(고창 고인돌 유적지)와 같은 넓은 공간을 **24시간 자동으로 순찰**할 수 있는 로봇 시스템 설계

- 운영자가 직접 이동하지 않고도
  - “특정 물체가 어디 있는지 확인”
  - “지정된 순찰 경로를 반복적으로 주행”
- 위 기능을 수행할 수 있는 **지능형 순찰 로봇 플랫폼** 구축

### 🧩 사용 플랫폼

- **OS / 미들웨어**
  - Ubuntu 22.04
  - ROS2 Humble
- **시뮬레이터**
  - Gazebo Classic
- **내비게이션 / 맵핑**
  - Navigation2 (nav2_bringup)
  - SLAM Toolbox
- **로봇**
  - TurtleBot3 Waffle (scale 2.0, 센서 확장)

### 📌 적용 시나리오

- 야간/무인 시간대에 로봇이 미리 설정된 **웨이포인트(waypoint)** 를 따라 자율 순찰
- LiDAR·카메라 정보는 RViz에서 **KITTI 스타일 포인트 클라우드**로 시각화
- (별도 Python 환경에서) YOLO + VLM(BLIP/CLIP 등)을 이용해  
  “사람 옆 노란 박스 찾아줘” 같은 **자연어 명령을 로봇 동작으로 연결**하는 구조를 설계

---

## 2. 시스템 구성

### 🤖 로봇 & 센서

- TurtleBot3 Waffle (scale 2.0)
- 2D LiDAR (`/scan`)
- 3D LiDAR (Velodyne, `/velodyne_points`)
- RGB 카메라, Depth 카메라
- Wheel encoder 기반 odometry (`/odom`)
- TF 트리: `/map → /odom → /base_link` 등

### 🧱 소프트웨어 스택

- **Gazebo world**
  - 고인돌 유적지 형태의 공원 환경  
    (팀원이 Blender로 제작한 맵)
  - `.sdf` world + 사용자 정의 모델
- **SLAM**
  - `slam_toolbox` 온라인 맵핑
- **자율주행**
  - `nav2_bringup` 기반 Navigation2
  - Python으로 작성된 **Waypoint 기반 순찰 노드**
- **가시화**
  - RViz2
  - `pointcloud_to_laserscan` + `PointCloud2` 디스플레이
  - `kitti_style_rviz.rviz` 설정으로  
    흰색 포인트 클라우드 + 긴 decay time 적용

---

## 3. 주요 기능

### 3-1. Phase 1 – 수동 탐색 & 웨이포인트 저장

- 키보드(teleop)로 로봇 직접 조종
- 특정 위치에서 `s` 키를 눌러 waypoint 저장
- 로봇의 `odom` / `map` 좌표를 읽어  
  **`waypoints.json`** 형태로 파일 저장
- 이후 Phase 2에서 이 파일을 그대로 사용

### 3-2. Phase 2 – 자동 순찰

- 저장된 `waypoints.json` 을 읽어들여 Navigation2 에 goal 연속 전송
- 순찰 경로를 **루프 형태**로 반복
- 장애물 발생 시 Nav2 replanning 기능으로 우회
- 실험 결과
  - 여러 차례 반복 순찰에서도 대부분 성공적으로 도착  
    (발표 자료 기준 약 **90% 이상 성공률**)

### 3-3. LiDAR 포인트 클라우드 시각화 (KITTI 스타일)

- 3D LiDAR 토픽 `/velodyne_points` 를 RViz에서 **빽빽한 흰색 점**으로 표현
- 설정 파일: `src/my_robot/config/kitti_style_rviz.rviz`
  - Style: `Flat Squares`
  - Size: 작게 (약 `0.02 ~ 0.03 m`)
  - Color: 흰색 (`FlatColor`)
  - Decay Time: 길게 설정하여, 지나간 경로도 “흔적”으로 남도록 구성
- Launch 파일: `src/my_robot/launch/kitti_visualization.launch.py`
  - `pointcloud_to_laserscan` 변환 노드
  - RViz2 실행 + 위 설정 자동 로딩

### 3-4. 카메라 & 센서 모니터링

- RGB / Depth 카메라 토픽을 `rqt_image_view` 로 실시간 확인
- LiDAR + 카메라 + 로봇 모델을 함께 띄워  
  **“로봇이 실제로 보는 장면”** 을 시각적으로 확인

---

## 4. 폴더 구조 (요약)

```text
my_robot_lim_last/
 ├─ src/
 │   └─ my_robot/
 │       ├─ launch/      # Gazebo, RViz, SLAM, Nav2, point cloud 시각화 launch
 │       ├─ config/      # Nav2, SLAM, RViz 설정 (kitti_style_rviz.rviz 포함)
 │       ├─ worlds/      # 고인돌 공원 Gazebo world 파일
 │       ├─ models/      # 사용자 정의 Gazebo 모델
 │       └─ scripts/     # waypoint 저장·자동 순찰 등 Python 노드
 └─ data/, map/, ...     # SLAM 결과, 로그 등
