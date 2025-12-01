"""
YOLO11n_empty_boost 모델 테스트
"""

from ultralytics import YOLO
import cv2
from pathlib import Path

base_dir = Path(__file__).parent
model_path = base_dir / 'yolo11n_empty_boost' / 'best.pt'
test_dir = base_dir / 'testImage'
output_dir = base_dir / 'yolo11n_empty_boost' / 'testImageResult'

output_dir.mkdir(exist_ok=True)

model = YOLO(str(model_path))

class_names = model.names
colors = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 165, 255)
}

print("=" * 70)
print("YOLO11n_empty_boost 모델 테스트 (Empty 강화 버전)")
print("=" * 70)
print(f"모델: {model_path}")
print("=" * 70)

test_images = sorted(test_dir.glob('*.png')) + sorted(test_dir.glob('*.jpg'))

print(f"\n총 {len(test_images)}개 이미지\n")

all_results = {}

for img_path in test_images:
    print(f"[{img_path.name}]")

    results = model(str(img_path), conf=0.25, verbose=False)
    img = cv2.imread(str(img_path))

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

    print(f"  탐지: {len(detections)}개")
    for det in detections:
        print(f"    - {det['class']} ({det['conf']:.0%})")

    all_results[img_path.name] = detections

    output_path = output_dir / f"{img_path.stem}_result.jpg"
    cv2.imwrite(str(output_path), img)

print(f"\n결과 저장: {output_dir}/")

print("\n" + "=" * 70)
print("전체 결과 요약")
print("=" * 70)

for img_name, dets in sorted(all_results.items()):
    print(f"\n{img_name}:")
    if len(dets) == 0:
        print("  탐지 없음")
    else:
        for det in dets:
            print(f"  - {det['class']} ({det['conf']:.0%})")

print("\n" + "=" * 70)
