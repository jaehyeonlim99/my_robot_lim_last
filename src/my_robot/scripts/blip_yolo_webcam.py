#!/usr/bin/env python3
import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image as PILImage
import cv2
import threading
import sys
import time


class BLIP_YOLO_Webcam:
    def __init__(self):
        print("🚀 BLIP + YOLO Node Starting...")

        # ✅ Device 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Using device: {self.device}")

        # ✅ BLIP 모델 (Caption + VQA)
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

        # ✅ YOLOv8 모델
        print("📦 Loading YOLOv8 model...")
        self.yolo = YOLO("yolov8n.pt")
        print("✅ YOLO model loaded successfully!")

        # ✅ 웹캠 연결
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")
            sys.exit(1)
        else:
            print("🎥 웹캠 연결 성공!")

        # ✅ 실행 상태
        self.running = True

        # ✅ 입력 스레드 시작
        self.user_input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.user_input_thread.start()

    def _input_loop(self):
        while self.running:
            try:
                question = input("\n💬 질문 입력 (예: Is there a chair? / 종료: exit): ").strip()
                if question.lower() in ["exit", "quit", "종료"]:
                    print("🛑 종료합니다.")
                    self.clean_exit()
                    return

                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ 프레임을 읽을 수 없습니다.")
                    continue

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(rgb_image)

                # YOLO 탐지
                detections = self.run_yolo(frame)

                # BLIP 질문/캡션
                if self.is_question_type(question):
                    answer = self.run_vqa(pil_image, question)
                else:
                    answer = self.run_captioning(pil_image)

                # 시각화
                self.visualize_result(frame, detections, answer)

            except KeyboardInterrupt:
                self.clean_exit()

    def run_yolo(self, frame):
        """YOLO 탐지"""
        results = self.yolo(frame, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = self.yolo.names[cls]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            detections.append({
                'label': label,
                'conf': conf,
                'coords': (x_center, y_center)
            })
        return detections

    def run_vqa(self, pil_image, question):
        """BLIP VQA"""
        print(f"🧠 [VQA] Q: {question}")
        inputs = self.vqa_processor(images=pil_image, text=question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.vqa_model.generate(**inputs, max_new_tokens=50)
        answer = self.vqa_processor.decode(output[0], skip_special_tokens=True)
        print(f"✅ A: {answer}")
        return answer

    def run_captioning(self, pil_image):
        """BLIP Captioning"""
        print("🖼️ [Captioning] Generating scene description...")
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.caption_model.generate(**inputs, max_new_tokens=50)
        caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
        print(f"📝 Caption: {caption}")
        return caption

    def visualize_result(self, frame, detections, text):
        """YOLO + BLIP 결과 시각화"""
        display_img = frame.copy()

        # 🔹 YOLO 바운딩박스
        for det in detections:
            label = det['label']
            conf = det['conf']
            (x_center, y_center) = det['coords']
            x1 = int(x_center - 40)
            y1 = int(y_center - 40)
            x2 = int(x_center + 40)
            y2 = int(y_center + 40)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(display_img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            print(f"📍 {label}: ({x_center:.1f}, {y_center:.1f})")

        # 🔹 BLIP 답변 표시
        cv2.rectangle(display_img, (10, 10), (850, 80), (0, 0, 0), -1)
        cv2.putText(display_img, f"{text[:100]}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('BLIP + YOLO Webcam Output', display_img)

        # 🔹 q 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.clean_exit()

    def is_question_type(self, text):
        """물음형 판단"""
        keywords = ["?", "있", "where", "what", "is there", "are there", "who"]
        return any(k in text.lower() for k in keywords)

    def clean_exit(self):
        """안전 종료"""
        print("🧹 종료 중...")
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    node = BLIP_YOLO_Webcam()
