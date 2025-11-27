"""
YOLO11 고급 추론 코드 - TTA (Test Time Augmentation)
이미지를 여러 방식으로 변환하여 추론 후 결과를 통합하여 더 정확한 탐지
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def flip_boxes_horizontal(boxes, img_width):
    """수평 반전된 이미지의 바운딩 박스를 원본 좌표로 변환"""
    flipped_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box['xyxy']
        new_x1 = img_width - x2
        new_x2 = img_width - x1
        flipped_boxes.append({
            'xyxy': [new_x1, y1, new_x2, y2],
            'conf': box['conf'],
            'cls': box['cls']
        })
    return flipped_boxes

def flip_boxes_vertical(boxes, img_height):
    """수직 반전된 이미지의 바운딩 박스를 원본 좌표로 변환"""
    flipped_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box['xyxy']
        new_y1 = img_height - y2
        new_y2 = img_height - y1
        flipped_boxes.append({
            'xyxy': [x1, new_y1, x2, new_y2],
            'conf': box['conf'],
            'cls': box['cls']
        })
    return flipped_boxes

def nms_boxes(boxes, iou_threshold=0.5):
    """Non-Maximum Suppression으로 중복 박스 제거"""
    if len(boxes) == 0:
        return []

    # 신뢰도 순으로 정렬
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)

    keep_boxes = []
    while boxes:
        best_box = boxes.pop(0)
        keep_boxes.append(best_box)

        boxes = [box for box in boxes if calculate_iou(best_box['xyxy'], box['xyxy']) < iou_threshold]

    return keep_boxes

def calculate_iou(box1, box2):
    """두 바운딩 박스의 IoU 계산"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 교집합 영역
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # 합집합 영역
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# 모델 로드
model_path = 'results/weights/best.pt'
model = YOLO(model_path)

print("=" * 70)
print("YOLO11 Advanced Inference - Test Time Augmentation (TTA)")
print("=" * 70)
print(f"Model: {model_path}")
print(f"Classes: {model.names}")
print("=" * 70)

# 테스트 이미지 경로
test_image_path = 'test.png'

if not os.path.exists(test_image_path):
    print(f"\n[ERROR] Test image not found: {test_image_path}")
    exit(1)

print(f"\nProcessing: {test_image_path}")
print("Applying Test Time Augmentation...")

# 원본 이미지 로드
original_img = cv2.imread(test_image_path)
height, width = original_img.shape[:2]

all_detections = []

# 1. 원본 이미지 추론
print("\n[1/6] Original image inference...")
results = model(original_img, conf=0.20, augment=True, verbose=False)
for box in results[0].boxes:
    all_detections.append({
        'xyxy': box.xyxy[0].tolist(),
        'conf': float(box.conf[0]),
        'cls': int(box.cls[0])
    })

# 2. 수평 반전 이미지 추론
print("[2/6] Horizontal flip inference...")
flipped_h = cv2.flip(original_img, 1)
results = model(flipped_h, conf=0.20, augment=True, verbose=False)
boxes_h = []
for box in results[0].boxes:
    boxes_h.append({
        'xyxy': box.xyxy[0].tolist(),
        'conf': float(box.conf[0]),
        'cls': int(box.cls[0])
    })
all_detections.extend(flip_boxes_horizontal(boxes_h, width))

# 3. 수직 반전 이미지 추론
print("[3/6] Vertical flip inference...")
flipped_v = cv2.flip(original_img, 0)
results = model(flipped_v, conf=0.20, augment=True, verbose=False)
boxes_v = []
for box in results[0].boxes:
    boxes_v.append({
        'xyxy': box.xyxy[0].tolist(),
        'conf': float(box.conf[0]),
        'cls': int(box.cls[0])
    })
all_detections.extend(flip_boxes_vertical(boxes_v, height))

# 4. 0.9배 스케일 추론
print("[4/6] Scale 0.9x inference...")
scaled_09 = cv2.resize(original_img, None, fx=0.9, fy=0.9)
results = model(scaled_09, conf=0.20, augment=True, verbose=False)
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    all_detections.append({
        'xyxy': [x1/0.9, y1/0.9, x2/0.9, y2/0.9],
        'conf': float(box.conf[0]),
        'cls': int(box.cls[0])
    })

# 5. 1.1배 스케일 추론
print("[5/6] Scale 1.1x inference...")
scaled_11 = cv2.resize(original_img, None, fx=1.1, fy=1.1)
results = model(scaled_11, conf=0.20, augment=True, verbose=False)
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    all_detections.append({
        'xyxy': [x1/1.1, y1/1.1, x2/1.1, y2/1.1],
        'conf': float(box.conf[0]),
        'cls': int(box.cls[0])
    })

# 6. 밝기 조정된 이미지 추론
print("[6/6] Brightness adjusted inference...")
hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255).astype(np.uint8)
bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
results = model(bright_img, conf=0.20, augment=True, verbose=False)
for box in results[0].boxes:
    all_detections.append({
        'xyxy': box.xyxy[0].tolist(),
        'conf': float(box.conf[0]),
        'cls': int(box.cls[0])
    })

print(f"\nTotal raw detections from all augmentations: {len(all_detections)}")

# NMS로 중복 제거
print("Applying Non-Maximum Suppression...")
final_detections = nms_boxes(all_detections, iou_threshold=0.4)

print(f"Final detections after NMS: {len(final_detections)}")

# 결과 이미지에 바운딩 박스 그리기
result_img = original_img.copy()
colors = {
    0: (0, 255, 0),    # fully_cart - 초록색
    1: (255, 0, 0),    # empty_cart - 파란색
    2: (0, 165, 255)   # combined_cart - 주황색
}

print("\n" + "=" * 70)
print(f"[FINAL RESULTS] - {len(final_detections)} objects detected")
print("=" * 70)

for i, det in enumerate(final_detections):
    x1, y1, x2, y2 = [int(coord) for coord in det['xyxy']]
    conf = det['conf']
    cls = det['cls']
    class_name = model.names[cls]

    # 바운딩 박스 그리기
    color = colors.get(cls, (255, 255, 255))
    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)

    # 라벨 텍스트
    label = f"{class_name} {conf:.2%}"
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(result_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    print(f"  [{i+1}] {class_name} - Confidence: {conf:.2%}")
    print(f"      Box: ({x1}, {y1}) -> ({x2}, {y2})")

# 결과 저장
output_path = 'test_result_tta.jpg'
cv2.imwrite(output_path, result_img)
print(f"\n[SAVED] Result saved to: {output_path}")

# 화면에 표시
result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(14, 10))
plt.imshow(result_img_rgb)
plt.axis('off')
plt.title(f'YOLO11 TTA Result - {len(final_detections)} objects detected (High Accuracy)',
          fontsize=16, pad=20, weight='bold')
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("TTA Inference completed!")
print("This method provides more accurate results by testing multiple augmentations.")
print("=" * 70)
