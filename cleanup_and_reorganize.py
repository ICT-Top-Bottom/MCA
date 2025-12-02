"""
labeledImage와 TEST/labeledImages 기준으로 images/labels 재정리
"""
from pathlib import Path
import shutil
import os

base_dir = Path(__file__).parent

# 디렉토리
labeled_dir = base_dir / 'labeledImage'
test_labeled_dir = base_dir / 'TEST' / 'labeledImages'
test_images_dir = base_dir / 'TEST' / 'empty_exif'
test_labels_dir = base_dir / 'TEST' / 'empty_exif_label'

main_images_dir = base_dir / 'images'
main_labels_dir = base_dir / 'labels'

# 임시 백업 디렉토리
temp_images_dir = base_dir / 'temp_images_backup'
temp_labels_dir = base_dir / 'temp_labels_backup'

print("=" * 70)
print("데이터셋 재정리")
print("=" * 70)

# 1. 현재 images와 labels를 임시 백업
print("\n1단계: 현재 데이터 백업 중...")
temp_images_dir.mkdir(exist_ok=True)
temp_labels_dir.mkdir(exist_ok=True)

for img_file in main_images_dir.glob('*'):
    if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
        shutil.move(str(img_file), str(temp_images_dir / img_file.name))

for label_file in main_labels_dir.glob('*.txt'):
    shutil.move(str(label_file), str(temp_labels_dir / label_file.name))

print("  백업 완료")

# 2. labeledImage에 있는 파일들을 images/labels로 복원
print("\n2단계: labeledImage 기준으로 복원 중...")

labeled_files = set()
for img_file in labeled_dir.glob('*.jpg'):
    labeled_files.add(img_file.stem)

restored_images = 0
restored_labels = 0

for stem in labeled_files:
    # 이미지 복원
    for ext in ['.jpg', '.png', '.jpeg']:
        src = temp_images_dir / f"{stem}{ext}"
        if src.exists():
            shutil.move(str(src), str(main_images_dir / src.name))
            restored_images += 1
            break

    # 라벨 복원
    src = temp_labels_dir / f"{stem}.txt"
    if src.exists():
        shutil.move(str(src), str(main_labels_dir / src.name))
        restored_labels += 1

print(f"  복원 완료: 이미지 {restored_images}개, 라벨 {restored_labels}개")

# 3. TEST/labeledImages에 있는 파일들 추가
print("\n3단계: TEST 데이터 추가 중...")

test_files = set()
for img_file in test_labeled_dir.glob('*.jpg'):
    test_files.add(img_file.stem)

added_images = 0
added_labels = 0

for stem in test_files:
    # 이미지 추가
    for ext in ['.jpg', '.png', '.jpeg']:
        src = test_images_dir / f"{stem}{ext}"
        if src.exists():
            dest = main_images_dir / src.name
            if not dest.exists():  # 중복 방지
                shutil.copy2(str(src), str(dest))
                added_images += 1
            break

    # 라벨 추가
    src = test_labels_dir / f"{stem}.txt"
    if src.exists():
        dest = main_labels_dir / src.name
        if not dest.exists():  # 중복 방지
            shutil.copy2(str(src), str(dest))
            added_labels += 1

print(f"  추가 완료: 이미지 {added_images}개, 라벨 {added_labels}개")

# 4. 최종 통계
print("\n" + "=" * 70)
print("최종 통계")
print("=" * 70)

from collections import defaultdict

class_counts = defaultdict(int)
class_names = {0: 'fully', 1: 'empty', 2: 'combined'}

for label_file in main_labels_dir.glob('*.txt'):
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                class_counts[class_id] += 1

total_images = len(list(main_images_dir.glob('*.jpg'))) + len(list(main_images_dir.glob('*.png')))
total_labels = len(list(main_labels_dir.glob('*.txt')))

print(f"\n총 이미지: {total_images}개")
print(f"총 라벨: {total_labels}개")

print("\n클래스별 객체 수:")
for class_id in sorted(class_counts.keys()):
    count = class_counts[class_id]
    print(f"  {class_names[class_id]}: {count}개")

total_objects = sum(class_counts.values())
print(f"\n총 객체: {total_objects}개")

# 클래스별 비율
if total_objects > 0:
    print("\n클래스별 비율:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = count / total_objects * 100
        print(f"  {class_names[class_id]}: {percentage:.1f}%")

# 불균형 비율
if len(class_counts) > 0:
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance = max_count / min_count if min_count > 0 else 0
    print(f"\n불균형 비율: {imbalance:.2f}:1")

# 5. 임시 백업 삭제
print("\n" + "=" * 70)
print("정리 중...")
print("=" * 70)

remaining_temp_images = len(list(temp_images_dir.glob('*')))
remaining_temp_labels = len(list(temp_labels_dir.glob('*')))

print(f"\n임시 폴더에 남은 파일:")
print(f"  이미지: {remaining_temp_images}개")
print(f"  라벨: {remaining_temp_labels}개")

print("\n임시 백업 폴더 삭제 중...")
shutil.rmtree(temp_images_dir)
shutil.rmtree(temp_labels_dir)
print("임시 폴더 삭제 완료")

print("\n" + "=" * 70)
print("재정리 완료!")
print("=" * 70)
