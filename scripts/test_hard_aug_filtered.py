"""
YOLO11s v6.2 Hard Augmentation - 비율 필터링 추론
conf=0.5 + 가로세로 비율 필터 + 크기 필터
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
from collections import Counter

MODEL_PATH = 'yolo11s_v6.2_hard_augmentation/best.pt'

print("="*70)
print("YOLO11s v6.2 Hard Aug - 비율 필터링 추론")
print("="*70)
print("필터 조건:")
print("  1. Confidence: 0.50 (높은 threshold)")
print("  2. 가로세로 비율: 0.3 ~ 2.5")
print("  3. 최소 크기: width >= 20px, height >= 20px")
print("="*70)

model = YOLO(MODEL_PATH)
print("✅ 모델 로드 완료\n")

# 클래스 정보
class_names = {0: 'combined', 1: 'empty', 2: 'fully'}
class_colors = {
    0: (255, 0, 0),    # combined: 파랑
    1: (0, 0, 255),    # empty: 빨강
    2: (0, 255, 0)     # fully: 초록
}

# 테스트 이미지
test_images = Path('testImage')
output_dir = Path('yolo11s_v6.2_hard_augmentation/filtered_test_results')
output_dir.mkdir(parents=True, exist_ok=True)

total_detections = Counter()
filtered_count = 0

print("테스트 시작 (conf=0.50 + 비율 필터링)...\n")

for img_path in sorted(list(test_images.glob('*.jpg')) + list(test_images.glob('*.png'))):
    print(f"처리 중: {img_path.name}")

    # 추론 (conf=0.5로 시작)
    results = model.predict(
        str(img_path),
        imgsz=640,
        conf=0.50,  # 높은 threshold ⭐
        iou=0.45,
        verbose=False,
        retina_masks=True
    )

    # 원본 이미지
    img = cv2.imread(str(img_path))
    annotated = img.copy()
    mask_overlay = img.copy()

    detections = Counter()
    current_filtered = 0

    for result in results:
        boxes = result.boxes
        masks = result.masks

        if boxes is not None and len(boxes) > 0:
            for idx, box in enumerate(boxes):
                # 박스 좌표 추출
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w = x2 - x1
                h = y2 - y1

                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # 가로세로 비율 계산
                aspect_ratio = w / h if h > 0 else 0

                # --- 필터링 로직 (Safety Net) ---

                # 1. 비율 필터링: 납작한 바닥(2.5+) 또는 얇은 기둥(0.3-) 제외
                if aspect_ratio > 2.5 or aspect_ratio < 0.3:
                    current_filtered += 1
                    continue

                # 2. 크기 필터링: 너무 작은 노이즈 제거
                if w < 20 or h < 20:
                    current_filtered += 1
                    continue

                # --- 통과한 진짜 객체만 처리 ---
                color = class_colors[cls]
                class_name = class_names[cls]

                # 세그멘테이션 마스크
                if masks is not None and idx < len(masks):
                    mask = masks[idx].data[0].cpu().numpy()
                    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    mask_bool = mask_resized > 0.5
                    mask_overlay[mask_bool] = color

                # 바운딩 박스
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                # 라벨 (비율 정보 포함)
                label = f"{class_name} {conf:.2f} (R:{aspect_ratio:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - 30), (x1 + tw + 10, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                detections[class_name] += 1
                total_detections[class_name] += 1

                print(f"    Detected: {class_name} (Conf: {conf:.2f}, Ratio: {aspect_ratio:.2f})")

    filtered_count += current_filtered

    # 마스크 합성
    annotated = cv2.addWeighted(annotated, 0.6, mask_overlay, 0.4, 0)

    # 저장
    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), annotated)

    print(f"  → 탐지: {dict(detections)} | 필터됨: {current_filtered}개\n")

print("="*70)
print("비율 필터링 추론 완료!")
print("="*70)
print(f"\n탐지 결과 (conf=0.50 + 필터링):")
for class_name in ['combined', 'empty', 'fully']:
    count = total_detections[class_name]
    print(f"  {class_name}: {count}개")

total = sum(total_detections.values())
print(f"\n전체: {total}개")
print(f"필터로 제거된 오탐: {filtered_count}개")
print(f"\n결과 저장: {output_dir}/")
print("="*70)
