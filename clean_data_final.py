"""
Data 폴더 최종 정리 스크립트
1. bbox 형식 파일 삭제
2. test → train 병합
3. 파일명 정리
4. Data_polygon 생성 (세그멘테이션 시각화)
"""
from pathlib import Path
import shutil
import cv2
import numpy as np

BASE_DIR = Path(r'c:\Users\24457\OneDrive\바탕 화면\MCA')
DATA_DIR = BASE_DIR / 'Data'
POLYGON_DIR = BASE_DIR / 'Data_polygon'

class_names = {0: 'combined', 1: 'empty', 2: 'fully'}
class_colors = {
    0: (255, 0, 0),    # combined: 파랑 (BGR)
    1: (0, 0, 255),    # empty: 빨강
    2: (0, 255, 0)     # fully: 초록
}

print("="*70)
print("Data 폴더 최종 정리")
print("="*70)

# Step 1: bbox 형식 라벨 삭제
print("\n[Step 1] bbox 형식 라벨 검사 및 삭제")
print("-"*70)

def is_bbox_format(label_file):
    """YOLO bbox 형식인지 확인 (class x y w h)"""
    try:
        with open(label_file, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                return False
            parts = first_line.split()
            return len(parts) == 5
    except:
        return False

bbox_files = []
for split in ['train', 'valid', 'test']:
    labels_dir = DATA_DIR / split / 'labels'
    if labels_dir.exists():
        for label_file in labels_dir.glob('*.txt'):
            if is_bbox_format(label_file):
                bbox_files.append(label_file)

if bbox_files:
    print(f"bbox 형식 라벨 발견: {len(bbox_files)}개")
    for f in bbox_files:
        # 이미지도 함께 삭제
        img_file = f.parent.parent / 'images' / f.stem.replace('_jpg', '.jpg')
        if not img_file.exists():
            img_file = f.parent.parent / 'images' / (f.stem + '.jpg')
        if not img_file.exists():
            img_file = f.parent.parent / 'images' / (f.stem + '.png')

        print(f"  삭제: {f.name}")
        f.unlink()
        if img_file.exists():
            img_file.unlink()

    print(f"완료: {len(bbox_files)}개 파일 삭제")
else:
    print("bbox 형식 라벨 없음 - 모두 segmentation 형식")

# Step 2: test → train 병합
print("\n[Step 2] test → train 병합")
print("-"*70)

test_dir = DATA_DIR / 'test'
train_dir = DATA_DIR / 'train'

if test_dir.exists():
    test_images = list((test_dir / 'images').glob('*'))
    print(f"test 이미지: {len(test_images)}개")

    moved_count = 0
    for img_file in test_images:
        lbl_file = test_dir / 'labels' / (img_file.stem + '.txt')

        # 이동
        shutil.move(str(img_file), str(train_dir / 'images' / img_file.name))
        if lbl_file.exists():
            shutil.move(str(lbl_file), str(train_dir / 'labels' / lbl_file.name))
        moved_count += 1

    # test 폴더 삭제
    shutil.rmtree(test_dir)
    print(f"완료: {moved_count}개 파일 이동, test 폴더 삭제")
else:
    print("test 폴더 없음")

# Step 3: 파일명 정리
print("\n[Step 3] 파일명 정리")
print("-"*70)

for split in ['train', 'valid']:
    images_dir = DATA_DIR / split / 'images'
    labels_dir = DATA_DIR / split / 'labels'

    if not images_dir.exists():
        continue

    image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    print(f"{split}: {len(image_files)}개")

    # 임시 폴더 생성
    temp_img_dir = DATA_DIR / split / 'images_temp'
    temp_lbl_dir = DATA_DIR / split / 'labels_temp'
    temp_img_dir.mkdir(exist_ok=True)
    temp_lbl_dir.mkdir(exist_ok=True)

    # 새 이름으로 복사
    for idx, img_file in enumerate(image_files, 1):
        new_name = f"{split}_{idx:04d}{img_file.suffix}"
        lbl_file = labels_dir / (img_file.stem + '.txt')

        shutil.copy(str(img_file), str(temp_img_dir / new_name))
        if lbl_file.exists():
            shutil.copy(str(lbl_file), str(temp_lbl_dir / f"{split}_{idx:04d}.txt"))

    # 원본 삭제 후 임시 폴더 이름 변경
    shutil.rmtree(images_dir)
    shutil.rmtree(labels_dir)
    temp_img_dir.rename(images_dir)
    temp_lbl_dir.rename(labels_dir)

    print(f"  → {split}_0001 ~ {split}_{len(image_files):04d}")

# Step 4: Data_polygon 생성
print("\n[Step 4] Data_polygon 생성 (세그멘테이션 시각화)")
print("-"*70)

if POLYGON_DIR.exists():
    shutil.rmtree(POLYGON_DIR)

POLYGON_DIR.mkdir()

total_visualized = 0

for split in ['train', 'valid']:
    images_dir = DATA_DIR / split / 'images'
    labels_dir = DATA_DIR / split / 'labels'

    if not images_dir.exists():
        continue

    # 출력 폴더 생성
    output_dir = POLYGON_DIR / split
    output_dir.mkdir(exist_ok=True)

    print(f"{split} 처리 중...")

    image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    processed = 0

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        label_file = labels_dir / (img_file.stem + '.txt')
        if not label_file.exists():
            continue

        # 라벨 읽기 및 그리기
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

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

        # 저장
        output_path = output_dir / img_file.name
        cv2.imwrite(str(output_path), img)
        processed += 1

    print(f"  → {processed}개 완료")
    total_visualized += processed

print(f"전체 {total_visualized}개 이미지 시각화 완료")

# 최종 통계
print("\n" + "="*70)
print("최종 결과")
print("="*70)

print("\n[Data/]")
for split in ['train', 'valid']:
    split_dir = DATA_DIR / split
    if split_dir.exists():
        img_count = len(list((split_dir / 'images').glob('*')))
        lbl_count = len(list((split_dir / 'labels').glob('*.txt')))
        print(f"  {split}/: 이미지 {img_count}개, 라벨 {lbl_count}개")

print("\n[Data_polygon/]")
for split in ['train', 'valid']:
    split_dir = POLYGON_DIR / split
    if split_dir.exists():
        img_count = len(list(split_dir.glob('*.jpg'))) + len(list(split_dir.glob('*.png')))
        print(f"  {split}/: {img_count}개 (라벨 없음)")

print("\n" + "="*70)
print("정리 완료!")
print("="*70)
