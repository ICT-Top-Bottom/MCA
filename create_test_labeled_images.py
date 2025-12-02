"""
TEST 폴더의 이미지에 라벨 시각화
"""
import cv2
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent
test_dir = base_dir / 'TEST'
images_dir = test_dir / 'empty_exif'
labels_dir = test_dir / 'empty_exif_label'
output_dir = test_dir / 'labeledImages'

output_dir.mkdir(exist_ok=True)

# 클래스 이름과 색상
class_names = {0: 'fully', 1: 'empty', 2: 'combined'}
colors = {
    0: (0, 255, 0),      # fully - 초록
    1: (255, 0, 0),      # empty - 파랑
    2: (0, 165, 255)     # combined - 주황
}

# 모든 라벨 파일 처리
label_files = sorted(labels_dir.glob('*.txt'))

print("=" * 70)
print("TEST 라벨 시각화 이미지 생성")
print("=" * 70)
print(f"총 {len(label_files)}개 파일 처리 중...\n")

processed = 0
skipped = 0

for label_file in label_files:
    stem = label_file.stem

    # 이미지 파일 찾기
    img_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        skipped += 1
        continue

    # 이미지 읽기 (한글 경로 처리)
    with open(img_path, 'rb') as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        skipped += 1
        continue

    h, w = img.shape[:2]

    # 라벨 읽기
    with open(label_file, 'r') as f:
        lines = f.readlines()

    # 바운딩 박스 그리기
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # YOLO 형식을 픽셀 좌표로 변환
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)

        # 바운딩 박스 그리기
        color = colors.get(class_id, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # 라벨 텍스트
        label = class_names[class_id]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 저장 (한글 경로 처리)
    output_path = output_dir / f"{stem}.jpg"
    is_success, buffer = cv2.imencode('.jpg', img)
    if is_success:
        with open(output_path, 'wb') as f:
            f.write(buffer)

    processed += 1
    if processed % 10 == 0:
        print(f"  처리 완료: {processed}/{len(label_files)}")

print(f"\n처리 완료: {processed}개")
print(f"건너뜀: {skipped}개")
print(f"\n저장 위치: {output_dir}/")
print("=" * 70)
