"""
삭제된 파일들을 deleteImage 폴더로 백업
"""
from pathlib import Path
import shutil

base_dir = Path(__file__).parent
labeled_dir = base_dir / 'labeledImage'
images_dir = base_dir / 'images'
labels_dir = base_dir / 'labels'

# deleteImage 폴더 구조 생성
delete_dir = base_dir / 'deleteImage'
delete_images_dir = delete_dir / 'images'
delete_labels_dir = delete_dir / 'labels'
delete_labeled_dir = delete_dir / 'labeledImages'

delete_images_dir.mkdir(parents=True, exist_ok=True)
delete_labels_dir.mkdir(parents=True, exist_ok=True)
delete_labeled_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("삭제된 파일 백업")
print("=" * 70)

# labeledImage에 남아있는 파일들
labeled_files = set()
for img_file in labeled_dir.glob('*.jpg'):
    labeled_files.add(img_file.stem)

print(f"labeledImage에 남은 파일: {len(labeled_files)}개\n")

# images에서 삭제된 파일 찾기
deleted_images = []
for img_file in images_dir.glob('*'):
    if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
        if img_file.stem not in labeled_files:
            deleted_images.append(img_file)

# labels에서 삭제된 파일 찾기
deleted_labels = []
for label_file in labels_dir.glob('*.txt'):
    if label_file.stem not in labeled_files:
        deleted_labels.append(label_file)

print(f"백업할 이미지: {len(deleted_images)}개")
print(f"백업할 라벨: {len(deleted_labels)}개")

if len(deleted_images) == 0 and len(deleted_labels) == 0:
    print("\n백업할 파일이 없습니다.")
    print("=" * 70)
    exit()

# 백업 실행
print("\n백업 중...")

backed_up_images = 0
backed_up_labels = 0
backed_up_labeled = 0

# images 백업
for img_file in deleted_images:
    try:
        shutil.copy2(img_file, delete_images_dir / img_file.name)
        backed_up_images += 1
    except Exception as e:
        print(f"  ❌ 백업 실패: {img_file.name} - {e}")

# labels 백업
for label_file in deleted_labels:
    try:
        shutil.copy2(label_file, delete_labels_dir / label_file.name)
        backed_up_labels += 1
    except Exception as e:
        print(f"  ❌ 백업 실패: {label_file.name} - {e}")

# labeledImage에서 삭제된 파일 찾기 (원본 649개와 비교)
import cv2
import numpy as np

# 원본 labels 폴더의 모든 파일
all_original_labels = set([f.stem for f in labels_dir.glob('*.txt')])

# labeledImage에 있는 파일
current_labeled = set([f.stem for f in labeled_dir.glob('*.jpg')])

# 삭제된 파일 = 원본에는 있었지만 현재 labeledImage에는 없는 것
deleted_from_labeled = all_original_labels - current_labeled

print(f"\nlabeledImage에서 삭제된 파일: {len(deleted_from_labeled)}개")

# 삭제된 파일들의 라벨 이미지를 다시 생성하여 백업
class_names = {0: 'fully', 1: 'empty', 2: 'combined'}
colors = {
    0: (0, 255, 0),      # fully - 초록
    1: (255, 0, 0),      # empty - 파랑
    2: (0, 165, 255)     # combined - 주황
}

for stem in deleted_from_labeled:
    # 이미지 파일 찾기
    img_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        continue

    # 라벨 파일 찾기
    label_file = labels_dir / f"{stem}.txt"
    if not label_file.exists():
        continue

    try:
        # 이미지 읽기 (한글 경로 처리)
        with open(img_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
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
        output_path = delete_labeled_dir / f"{stem}.jpg"
        is_success, buffer = cv2.imencode('.jpg', img)
        if is_success:
            with open(output_path, 'wb') as f:
                f.write(buffer)
            backed_up_labeled += 1

    except Exception as e:
        print(f"  ❌ labeledImage 생성 실패: {stem} - {e}")

print(f"\n백업 완료:")
print(f"  images: {backed_up_images}개")
print(f"  labels: {backed_up_labels}개")
print(f"  labeledImages: {backed_up_labeled}개")

print(f"\n백업 위치: {delete_dir}/")
print("=" * 70)
