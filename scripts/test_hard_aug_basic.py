"""
YOLO11s v6.2 Hard Augmentation - 기본 추론
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
from collections import Counter

MODEL_PATH = 'yolo11s_v6.2_hard_augmentation/best.pt'

print("="*70)
print("YOLO11s v6.2 Hard Augmentation - 기본 추론")
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
output_dir = Path('yolo11s_v6.2_hard_augmentation/basic_test_results')
output_dir.mkdir(parents=True, exist_ok=True)

total_detections = Counter()

print("테스트 시작 (conf=0.40)...\n")

for img_path in sorted(list(test_images.glob('*.jpg')) + list(test_images.glob('*.png'))):
    print(f"처리 중: {img_path.name}")

    # 추론
    results = model.predict(
        str(img_path),
        conf=0.40,
        iou=0.45,
        verbose=False,
        retina_masks=True
    )

    # 원본 이미지
    img = cv2.imread(str(img_path))
    annotated = img.copy()
    mask_overlay = img.copy()

    detections = Counter()

    for result in results:
        boxes = result.boxes
        masks = result.masks

        if boxes is not None and len(boxes) > 0:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])

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

                # 라벨
                label = f"{class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                detections[class_name] += 1
                total_detections[class_name] += 1

    # 마스크 합성
    annotated = cv2.addWeighted(annotated, 0.6, mask_overlay, 0.4, 0)

    # 저장
    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), annotated)

    print(f"  → {dict(detections)}")

print("\n" + "="*70)
print("기본 추론 완료!")
print("="*70)
print(f"\n탐지 결과 (conf=0.40):")
for class_name in ['combined', 'empty', 'fully']:
    count = total_detections[class_name]
    print(f"  {class_name}: {count}개")

total = sum(total_detections.values())
print(f"\n전체: {total}개")
print(f"\n결과 저장: {output_dir}/")
print("="*70)
