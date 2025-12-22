"""
라벨 없는 fully 이미지 제거 및 번호 재정렬
"""
import os
from pathlib import Path
import shutil

images_dir = Path('images')
labels_dir = Path('labels')

print("=" * 70)
print("라벨 없는 fully 이미지 제거 및 재정렬")
print("=" * 70)

# 라벨 없는 fully 번호 찾기
fully_images = set(range(174))
fully_labels = {int(f.stem.split('_')[1]) for f in labels_dir.glob('fully_*.txt')}
missing = sorted(fully_images - fully_labels)

print(f"\n라벨 없는 fully 이미지: {len(missing)}개")
print(f"번호: {missing}")

# 라벨 없는 이미지 삭제
print("\n삭제 중...")
deleted_count = 0
for num in missing:
    img_path = images_dir / f"fully_{num}.jpg"
    if img_path.exists():
        os.remove(img_path)
        print(f"  삭제: fully_{num}.jpg")
        deleted_count += 1

print(f"\n총 {deleted_count}개 이미지 삭제 완료")

# fully 클래스만 0번부터 재정렬
print("\nfully 클래스 재정렬 중...")

# 라벨이 있는 fully 파일만 수집
fully_files = []
for img_path in sorted(images_dir.glob('fully_*.jpg')):
    label_path = labels_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        fully_files.append((img_path, label_path))

print(f"라벨 있는 fully 파일: {len(fully_files)}개")

# 임시 이름으로 변경
temp_mappings = []
for idx, (img_path, label_path) in enumerate(fully_files):
    temp_img = images_dir / f"temp_fully_{idx}.jpg"
    temp_label = labels_dir / f"temp_fully_{idx}.txt"

    shutil.move(str(img_path), str(temp_img))
    shutil.move(str(label_path), str(temp_label))
    temp_mappings.append((temp_img, temp_label, idx))

# 최종 이름으로 변경
for temp_img, temp_label, idx in temp_mappings:
    final_img = images_dir / f"fully_{idx}.jpg"
    final_label = labels_dir / f"fully_{idx}.txt"

    shutil.move(str(temp_img), str(final_img))
    shutil.move(str(temp_label), str(final_label))

    if idx < 5 or idx >= len(temp_mappings) - 2:
        print(f"  [{idx:3d}] fully_{idx}.jpg")
    elif idx == 5:
        print(f"  ...")

print("\n" + "=" * 70)
print("작업 완료")
print("=" * 70)

# 최종 확인
print("\n최종 파일 개수:")
for class_name in ['fully', 'empty', 'combined']:
    img_count = len(list(images_dir.glob(f'{class_name}_*.jpg')))
    label_count = len(list(labels_dir.glob(f'{class_name}_*.txt')))
    print(f"  {class_name}: 이미지 {img_count}개, 라벨 {label_count}개")

total_images = len(list(images_dir.glob('*.jpg')))
total_labels = len(list(labels_dir.glob('*.txt')))
print(f"\n전체: 이미지 {total_images}개, 라벨 {total_labels}개")

if total_images == total_labels:
    print("\n모든 이미지-라벨 쌍이 일치합니다!")
else:
    print(f"\n경고: 이미지-라벨 불일치 ({total_images - total_labels}개 차이)")
