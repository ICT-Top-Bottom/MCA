"""
YOLO11 모델 추론 코드
학습된 모델을 사용하여 이미지에서 객체 탐지 및 바운딩 박스 표시
"""

from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

# 모델 로드
model_path = 'results/weights/best.pt'
model = YOLO(model_path)

print("=" * 60)
print("YOLO11 Cart Detection - Inference")
print("=" * 60)
print(f"Model: {model_path}")
print(f"Classes: {model.names}")
print("=" * 60)

# 테스트 이미지 경로
test_image_path = 'test.png'

# 이미지 존재 확인
if not os.path.exists(test_image_path):
    print(f"\n[ERROR] Test image not found: {test_image_path}")
    print("Please place 'test.jpg' in the root folder.")
    exit(1)

print(f"\nProcessing: {test_image_path}")

# 추론 실행
results = model(test_image_path, conf=0.25)  # confidence threshold 0.25

# 결과 처리
for idx, result in enumerate(results):
    # 바운딩 박스가 그려진 이미지
    annotated_image = result.plot()

    # 결과 저장
    output_path = 'test_result.jpg'
    cv2.imwrite(output_path, annotated_image)

    # 이미지 화면에 표시
    # BGR to RGB 변환 (OpenCV는 BGR, matplotlib은 RGB 사용)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image_rgb)
    plt.axis('off')
    plt.title(f'YOLO11 Detection Result - {len(result.boxes)} objects detected', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

    # 탐지된 객체 정보 출력
    boxes = result.boxes
    print(f"\n[RESULTS]")
    print(f"Detected objects: {len(boxes)}")

    if len(boxes) > 0:
        print("\nDetection details:")
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]

            # 바운딩 박스 좌표 (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(f"  [{i+1}] {class_name} - Confidence: {conf:.2%}")
            print(f"      Box: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
    else:
        print("\nNo objects detected.")

    print(f"\n[SAVED] Result saved to: {output_path}")

print("\n" + "=" * 60)
print("Inference completed!")
print("=" * 60)
