#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import threading
import sys
import time

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image as PILImage
import cv2

# Ultralytics YOLOv8
from ultralytics import YOLO


class BLIPYOLONode(Node):
    def __init__(self):
        super().__init__('blip_yolo_node')
        self.get_logger().info("🚀 BLIP + YOLO node starting...")

        # ---------- Device ----------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"📱 Using device: {self.device}")

        # ---------- BLIP (VQA) ----------
        self.get_logger().info("📦 Loading BLIP-VQA...")
        self.vqa_model_name = "Salesforce/blip-vqa-base"
        self.blip_processor = BlipProcessor.from_pretrained(self.vqa_model_name)
        self.blip_model = BlipForQuestionAnswering.from_pretrained(
            self.vqa_model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        ).to(self.device)
        self.get_logger().info("✅ BLIP ready")

        # ---------- YOLO ----------
        self.get_logger().info("📦 Loading YOLOv8 (n)...")
        self.yolo = YOLO("yolov8n.pt")  # 가벼운 기본 모델
        # 결과물 자동 저장 방지
        try:
            self.yolo.overrides['save'] = False
        except Exception:
            pass
        self.get_logger().info("✅ YOLO ready")

        # ---------- ROS image subscription ----------
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_count = 0

        self.subscription = self.create_subscription(
            Image, '/camera_left/image_raw', self.image_callback, 10
        )
        # 첫 프레임이 오기 전에 입력 프롬프트가 먼저 출력되도록 약간의 지연
        self.get_logger().info("🎥 Subscribed to /camera_left/image_raw")
        sys.stdout.flush()
        time.sleep(0.2)

        # ---------- UI / State ----------
        self.running = True
        self.last_answer_text = ""
        self.win_name = "BLIP + YOLO"

        # 입력 스레드 시작
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        # 주기적인 화면 갱신(프롬프트 대기 중에도 화면 유지)
        self.timer = self.create_timer(0.2, self._refresh_window)

    # -------------------- ROS callbacks --------------------
    def image_callback(self, msg: Image):
        try:
            # encoding이 rgb8이면 BGR로 바꿔야 OpenCV 표준 시각화와 일치
            if msg.encoding.lower().startswith('rgb'):
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            self.latest_frame = cv_img
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    # -------------------- Input / Inference --------------------
    def _input_loop(self):
        """터미널에서 질문을 입력받고, 한 번 추론을 수행한다."""
        while self.running:
            try:
                q = input("\n💬 질문 입력 (예: Is there a chair? / 종료: exit): ").strip()
                if q.lower() in ["exit", "quit", "종료"]:
                    self.get_logger().info("🛑 종료합니다.")
                    self.safe_shutdown()
                    return

                if self.latest_frame is None:
                    print("⚠️ 아직 카메라 프레임이 없습니다. 잠시 후 다시 시도하세요.")
                    continue

                # 1) BLIP VQA
                blip_answer = self.run_blip_vqa(self.latest_frame, q)

                # 2) YOLO detection
                detections = self.run_yolo(self.latest_frame)

                # 3) 매칭(선택): 질문 키워드가 있으면 간단 매칭 시도
                matched = self.simple_match(q, detections)

                # 4) 시각화 텍스트 구성
                if len(detections) == 0:
                    det_text = "❌ No object detected in image"
                else:
                    det_text = "✅ Detected: " + ", ".join(
                        [f"{d['class']}({d['confidence']:.2f})" for d in detections]
                    )

                self.last_answer_text = (
                    f"Q: {q}\n"
                    f"💭 BLIP Answer: {blip_answer}\n"
                    f"{det_text}\n"
                    f"{'🎯 Matched!' if matched else ''}"
                )

                # 한 번 그려주기
                self.visualize(self.latest_frame, q, blip_answer, detections)

            except EOFError:
                self.safe_shutdown()
                return
            except KeyboardInterrupt:
                self.safe_shutdown()
                return

    # -------------------- Models --------------------
    def run_blip_vqa(self, bgr_image, question: str) -> str:
        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        inputs = self.blip_processor(images=pil_img, text=question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.blip_model.generate(**inputs, max_new_tokens=40)
        answer = self.blip_processor.decode(output_ids[0], skip_special_tokens=True)
        self.get_logger().info(f"? Question: {question}")
        self.get_logger().info(f"💬 BLIP Answer: {answer}")
        return answer

    def run_yolo(self, bgr_image):
        """Ultralytics 최신 API 방식으로 박스 파싱."""
        results = self.yolo(bgr_image, verbose=False)[0]
        detections = []
        try:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = results.names[cls_id] if hasattr(results, 'names') else str(cls_id)
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                })
        except Exception as e:
            self.get_logger().warn(f"YOLO parsing warning: {e}")
        return detections

    # 간단 매칭(질문에 등장하는 단어가 탐지 클래스에 포함되면 매치로 간주)
    def simple_match(self, question: str, detections) -> bool:
        q = question.lower()
        for det in detections:
            if det["class"].lower() in q:
                return True
        return False

    # -------------------- Visualization --------------------
    def visualize(self, frame_bgr, question: str, blip_answer: str, detections):
        img = frame_bgr.copy()

        # 상단 텍스트 박스
        top_h = 90
        cv2.rectangle(img, (10, 10), (10 + 800, 10 + top_h), (0, 0, 0), -1)
        cv2.putText(img, f"Q: {question}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(img, f"A: {blip_answer}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # YOLO 박스
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 200, 255), 2)
            cv2.circle(img, (int(det["center"][0]), int(det["center"][1])), 4, (0, 255, 0), -1)
            cv2.putText(img, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2)

        cv2.imshow(self.win_name, img)
        # 'q'로 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("🛑 'q' pressed. Shutting down.")
            self.safe_shutdown()

    def _refresh_window(self):
        """프레임이 들어오면 마지막 답변 텍스트라도 띄워 사용자가 상태를 볼 수 있게 함."""
        if self.latest_frame is None:
            return
        # 최근 정보가 있으면 간단히 상단만 표시해 유지
        img = self.latest_frame.copy()
        if self.last_answer_text:
            block_w, block_h = 900, 110
            cv2.rectangle(img, (10, 10), (10 + block_w, 10 + block_h), (0, 0, 0), -1)
            for i, line in enumerate(self.last_answer_text.split("\n")[:3]):
                cv2.putText(img, line, (20, 45 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if i == 1 else (255, 255, 255), 2)
        cv2.imshow(self.win_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.safe_shutdown()

    # -------------------- Shutdown --------------------
    def safe_shutdown(self):
        if not self.running:
            return
        self.running = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        # 입력 스레드가 살아있다면 종료 유도
        try:
            sys.stdout.flush()
        except Exception:
            pass
        # 프로세스 종료는 메인에서 맡김


def main(args=None):
    rclpy.init(args=args)
    node = BLIPYOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.safe_shutdown()
    finally:
        # 중복 종료 방지
        if rclpy.ok():
            rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
