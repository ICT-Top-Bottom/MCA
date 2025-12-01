"""
yolo11n_240 모델로 모든 테스트 이미지 추론 (배치)
결과를 저장하여 성능 비교
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np

# 경로 설정
base_dir = Path(__file__).parent
model_path = base_dir / 'yolo11n_240' / 'results' / 'weights' / 'best.pt'
test_dir = base_dir / 'testImage'
output_dir = base_dir / 'yolo11n_240_test_results'

output_dir.mkdir(exist_ok=True)

# 모델 로드
model = YOLO(str(model_path))

class_names = model.names
colors = {
    0: (0, 255, 0),    # fully - 초록
    1: (255, 0, 0),    # empty - 파란
    2: (0, 165, 255)   # combined - 주황
}

print("=" * 70)
print("YOLO11n_240 모델 테스트 (전체 이미지)")
print("=" * 70)
print(f"모델: {model_path}")
print(f"테스트 이미지 폴더: {test_dir}")
print("=" * 70)

# 테스트 이미지 찾기
test_images = sorted(test_dir.glob('*.png')) + sorted(test_dir.glob('*.jpg'))

print(f"\n총 {len(test_images)}개 이미지 처리 중...\n")

# 각 이미지 추론
for img_path in test_images:
    print(f"[{img_path.name}]")

    # 추론
    results = model(str(img_path), conf=0.25, verbose=False)

    # 이미지 로드
    img = cv2.imread(str(img_path))

    # 바운딩 박스 그리기
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        detections.append({
            'class': class_names[cls],
            'conf': conf,
            'bbox': (x1, y1, x2, y2)
        })

        color = colors.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        label = f"{class_names[cls]} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 결과 출력
    print(f"  탐지: {len(detections)}개")
    for det in detections:
        print(f"    - {det['class']} ({det['conf']:.0%})")

    # 저장
    output_path = output_dir / f"{img_path.stem}_result.jpg"
    cv2.imwrite(str(output_path), img)

print(f"\n결과 저장: {output_dir}/")
print("=" * 70)
