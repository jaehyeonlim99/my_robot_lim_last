#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image as PILImage
import cv2
import threading
import sys
import time


class BLIPHybridNode(Node):
    def __init__(self):
        super().__init__('blip_hybrid_node')
        self.get_logger().info("🚀 BLIP Hybrid Node Starting (Caption + VQA)...")

        # ✅ GPU 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"📱 Using device: {self.device}")

        # ✅ 모델 불러오기
        self.caption_model_name = "Salesforce/blip-image-captioning-large"
        self.vqa_model_name = "Salesforce/blip-vqa-base"

        self.caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            self.caption_model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        ).to(self.device)

        self.vqa_processor = BlipProcessor.from_pretrained(self.vqa_model_name)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(
            self.vqa_model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        ).to(self.device)

        self.get_logger().info("✅ Both models loaded successfully!")

        # ✅ ROS 이미지 구독
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera_left/image_raw', self.image_callback, 10
        )

        self.latest_frame = None
        self.frame_count = 0
        self.running = True

        # 🎥 Subscribed 로그 → flush 처리 (출력 순서 보정)
        self.get_logger().info("🎥 Subscribed to /camera_left/image_raw")
        sys.stdout.flush()
        time.sleep(0.2)

        # ✅ 사용자 입력 스레드 시작
        self.user_input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.user_input_thread.start()

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def _input_loop(self):
        """💬 입력 기반 추론 (타이머 없이, 입력 시만 실행)"""
        while self.running:
            try:
                question = input("\n💬 질문을 입력하세요 (예: Is there a red apple? / 종료: exit): ").strip()
                sys.stdout.flush()

                if question.lower() in ["exit", "quit", "종료"]:
                    self.get_logger().info("🛑 종료 명령 감지 — 프로그램을 종료합니다.")
                    self.running = False
                    cv2.destroyAllWindows()
                    try:
                        rclpy.try_shutdown()  # 중복 shutdown 방지
                    except Exception:
                        pass
                    sys.exit(0)

                if self.latest_frame is None:
                    print("⚠️ 아직 카메라 프레임을 받지 못했습니다. 잠시 후 다시 시도하세요.")
                    continue

                rgb_image = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(rgb_image)

                # 물음형이면 VQA, 아니면 Caption
                if self.is_question_type(question):
                    answer = self.run_vqa(pil_image, question)
                else:
                    answer = self.run_captioning(pil_image)

                self.visualize_result(self.latest_frame, f"{answer}")

            except EOFError:
                break
            except KeyboardInterrupt:
                self.clean_exit()

    def clean_exit(self):
        """🧹 종료 처리"""
        self.get_logger().info("🧹 프로그램을 종료합니다.")
        self.running = False
        cv2.destroyAllWindows()
        try:
            rclpy.try_shutdown()
        except Exception:
            pass
        sys.exit(0)

    def is_question_type(self, text):
        """문장이 물음형인지 판단"""
        keywords = ["?", "있", "where", "what", "is there", "are there", "who"]
        return any(k in text.lower() for k in keywords)

    def run_vqa(self, pil_image, question):
        """BLIP VQA 실행"""
        self.get_logger().info(f"🧠 [VQA] Q: {question}")
        inputs = self.vqa_processor(images=pil_image, text=question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.vqa_model.generate(**inputs, max_new_tokens=50)
        answer = self.vqa_processor.decode(output[0], skip_special_tokens=True)
        self.get_logger().info(f"✅ A: {answer}")
        return f"Q: {question} | A: {answer}"

    def run_captioning(self, pil_image):
        """BLIP Captioning 실행"""
        self.get_logger().info("🖼️ [Captioning] Generating scene description...")
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.caption_model.generate(**inputs, max_new_tokens=50)
        caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
        self.get_logger().info(f"📝 Caption: {caption}")
        return caption

    def visualize_result(self, image, text):
        """결과 시각화"""
        display_img = image.copy()
        cv2.rectangle(display_img, (10, 10), (850, 80), (0, 0, 0), -1)
        cv2.putText(display_img, text[:100], (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('BLIP Hybrid Output', display_img)

        # 🔹 q 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.clean_exit()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = BLIPHybridNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.clean_exit()
    finally:
        cv2.destroyAllWindows()
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
