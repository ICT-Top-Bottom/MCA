"""
모든 이미지에 라벨 바운딩 박스 시각화
결과를 LAST 폴더에 저장
"""
import cv2
from pathlib import Path

images_dir = Path('images')
labels_dir = Path('labels')
output_dir = Path('LAST')
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("라벨 바운딩 박스 시각화")
print("=" * 70)

# 클래스별 색상
class_colors = {
    0: (0, 255, 0),    # fully: 초록
    1: (0, 0, 255),    # empty: 빨강
    2: (255, 0, 0)     # combined: 파랑
}

class_names = {
    0: 'fully',
    1: 'empty',
    2: 'combined'
}

# 모든 이미지 처리
image_files = sorted(images_dir.glob('*.jpg'))
print(f"\n총 {len(image_files)}개 이미지 처리 중...\n")

processed = 0
total_boxes = 0

for img_path in image_files:
    # 라벨 파일 경로
    label_path = labels_dir / f"{img_path.stem}.txt"

    if not label_path.exists():
        print(f"  [WARNING] 라벨 없음: {img_path.name}")
        continue

    # 이미지 읽기
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [ERROR] 이미지 로드 실패: {img_path.name}")
        continue

    h, w = img.shape[:2]
    box_count = 0

    # 라벨 읽기 및 그리기
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # YOLO 형식 -> 픽셀 좌표 변환
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # 바운딩 박스 그리기
            color = class_colors[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 라벨 텍스트
            label = class_names[cls]

            # 배경
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), color, -1)

            # 텍스트
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            box_count += 1
            total_boxes += 1

    # 저장
    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), img)

    processed += 1

    # 진행상황 출력 (일부만)
    if processed <= 10 or processed % 50 == 0 or processed == len(image_files):
        print(f"  [{processed:3d}/{len(image_files)}] {img_path.name} - {box_count}개 박스")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
print(f"\n처리된 이미지: {processed}개")
print(f"총 바운딩 박스: {total_boxes}개")
print(f"\n결과 저장: {output_dir}/")
print("\n색상:")
print("  - 초록색: fully")
print("  - 빨간색: empty")
print("  - 파란색: combined")
