"""
Data_polygon 폴더 생성: 세그멘테이션 마스크가 시각화된 이미지만 저장
"""
from pathlib import Path
import cv2
import numpy as np
import shutil

BASE_DIR = Path(r'c:\Users\24457\OneDrive\바탕 화면\MCA')
DATA_DIR = BASE_DIR / 'Data'
POLYGON_DIR = BASE_DIR / 'Data_polygon'

# 클래스 정보
class_names = {0: 'combined', 1: 'empty', 2: 'fully'}
class_colors = {
    0: (255, 0, 0),    # combined: 파랑 (BGR)
    1: (0, 0, 255),    # empty: 빨강
    2: (0, 255, 0)     # fully: 초록
}

print("="*70)
print("Data_polygon 생성 (세그멘테이션 시각화)")
print("="*70)

# Data_polygon 폴더 생성
if POLYGON_DIR.exists():
    shutil.rmtree(POLYGON_DIR)

POLYGON_DIR.mkdir()

total_processed = 0

for split in ['train', 'valid']:
    data_split = DATA_DIR / split

    if not data_split.exists():
        continue

    images_dir = data_split / 'images'
    labels_dir = data_split / 'labels'

    if not images_dir.exists() or not labels_dir.exists():
        continue

    # 출력 폴더 생성
    output_dir = POLYGON_DIR / split
    output_dir.mkdir(exist_ok=True)

    print(f"\n[{split}] 처리 중...")

    image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    processed = 0

    for img_file in image_files:
        # 이미지 로드
        img = cv2.imread(str(img_file))

        if img is None:
            print(f"  경고: {img_file.name} 읽기 실패")
            continue

        # 라벨 파일 찾기
        label_file = labels_dir / (img_file.stem + '.txt')

        if not label_file.exists():
            print(f"  경고: {img_file.name}의 라벨 없음")
            continue

        # 라벨 읽기 및 그리기
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                # 세그멘테이션 형식인지 확인
                if len(coords) <= 4:
                    continue

                # 절대 좌표로 변환
                h, w = img.shape[:2]
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)

                # 폴리곤 그리기
                color = class_colors.get(class_id, (128, 128, 128))
                cv2.polylines(img, [points], isClosed=True, color=color, thickness=3)

                # 바운딩 박스 계산
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # 클래스명 표시
                class_name = class_names.get(class_id, f'class_{class_id}')

                # 텍스트 배경
                (text_w, text_h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, (x_min, y_min - 30), (x_min + text_w + 10, y_min), color, -1)

                # 텍스트
                cv2.putText(img, class_name, (x_min + 5, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 저장 (images 폴더 없이 바로 split 폴더에)
        output_path = output_dir / f"{split}_{processed+1:04d}.jpg"
        cv2.imwrite(str(output_path), img)
        processed += 1

    print(f"  → {processed}개 이미지 처리 완료")
    total_processed += processed

print("\n" + "="*70)
print(f"전체 {total_processed}개 이미지 시각화 완료!")
print("Data_polygon/ 폴더에 이미지만 저장됨 (라벨 파일 없음)")
print("="*70)

# 최종 확인
print("\n[Data_polygon 구조]")
for split in ['train', 'valid']:
    split_dir = POLYGON_DIR / split
    if split_dir.exists():
        img_count = len(list(split_dir.glob('*.jpg'))) + len(list(split_dir.glob('*.png')))
        print(f"  {split}/: {img_count}개")
