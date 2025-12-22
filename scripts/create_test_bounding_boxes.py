"""
images 폴더의 BOOK 이미지들에 바운딩 박스를 그려서
testBoundingBox 폴더에 저장
"""
import cv2
import numpy as np
from pathlib import Path
import shutil

# 경로 설정
images_dir = Path('images')
labels_dir = Path('labels')
output_dir = Path('testBoundingBox')

# 출력 폴더 생성
output_dir.mkdir(exist_ok=True)

# 클래스 정보
class_names = {
    0: 'fully',
    1: 'empty',
    2: 'combined'
}

class_colors = {
    0: (0, 255, 0),      # fully: 초록
    1: (0, 0, 255),      # empty: 빨강
    2: (255, 165, 0)     # combined: 주황
}

print("=" * 70)
print("바운딩 박스 시각화")
print("=" * 70)
print(f"Input: {images_dir}")
print(f"Output: {output_dir}")
print(f"\nClass colors:")
print(f"  - fully (0): Green")
print(f"  - empty (1): Red")
print(f"  - combined (2): Orange")

# BOOK이 포함된 이미지 파일 찾기
book_images = sorted(images_dir.glob('*BOOK*.jpg'))
print(f"\n총 {len(book_images)}개의 BOOK 이미지 발견")

if len(book_images) == 0:
    print("[WARNING] BOOK 이미지가 없습니다!")
    exit(1)

# 각 이미지 처리
processed = 0
for img_path in book_images:
    # 라벨 파일 경로 (BOOK 부분 제거)
    # fully_138-BOOK-7F1IFULU9P.jpg -> fully_138.txt
    base_name = img_path.stem.split('-BOOK-')[0]
    label_path = labels_dir / f"{base_name}.txt"

    if not label_path.exists():
        print(f"[WARNING] Label not found: {img_path.name}")
        continue

    # 이미지 읽기 (한글 경로 처리)
    with open(img_path, 'rb') as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        print(f"[WARNING] Failed to read: {img_path.name}")
        continue

    h, w = img.shape[:2]

    # 라벨 읽기 및 바운딩 박스 그리기
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # YOLO 형식: class_id x_center y_center width height (normalized)
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # 픽셀 좌표로 변환
            x_center_px = int(x_center * w)
            y_center_px = int(y_center * h)
            box_width_px = int(box_width * w)
            box_height_px = int(box_height * h)

            # 좌상단, 우하단 좌표 계산
            x1 = int(x_center_px - box_width_px / 2)
            y1 = int(y_center_px - box_height_px / 2)
            x2 = int(x_center_px + box_width_px / 2)
            y2 = int(y_center_px + box_height_px / 2)

            # 바운딩 박스 그리기
            color = class_colors[class_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # 클래스 이름 텍스트
            label_text = class_names[class_id]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # 텍스트 배경
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)

            # 텍스트
            cv2.putText(img, label_text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)

    # 저장 (한글 경로 처리)
    output_path = output_dir / img_path.name
    is_success, buffer = cv2.imencode('.jpg', img)
    if is_success:
        with open(output_path, 'wb') as f:
            f.write(buffer)
        processed += 1
        print(f"[OK] {img_path.name}")
    else:
        print(f"[WARNING] Failed to save: {img_path.name}")

print("\n" + "=" * 70)
print(f"완료! {processed}개 이미지 처리됨")
print(f"저장 위치: {output_dir}/")
print("=" * 70)
