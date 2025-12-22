"""
labeledImage에서 삭제된 파일들을 images와 labels에서도 삭제
"""
from pathlib import Path
import os

base_dir = Path(__file__).parent
labeled_dir = base_dir / 'labeledImage'
images_dir = base_dir / 'images'
labels_dir = base_dir / 'labels'

print("=" * 70)
print("원본 파일 동기화 삭제")
print("=" * 70)

# labeledImage에 남아있는 파일들
labeled_files = set()
for img_file in labeled_dir.glob('*.jpg'):
    labeled_files.add(img_file.stem)

print(f"labeledImage에 남은 파일: {len(labeled_files)}개\n")

# images와 labels에서 삭제할 파일 찾기
to_delete_images = []
to_delete_labels = []

# images 확인
for img_file in images_dir.glob('*'):
    if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
        if img_file.stem not in labeled_files:
            to_delete_images.append(img_file)

# labels 확인
for label_file in labels_dir.glob('*.txt'):
    if label_file.stem not in labeled_files:
        to_delete_labels.append(label_file)

print(f"삭제할 이미지: {len(to_delete_images)}개")
print(f"삭제할 라벨: {len(to_delete_labels)}개")

if len(to_delete_images) == 0 and len(to_delete_labels) == 0:
    print("\n삭제할 파일이 없습니다.")
    print("=" * 70)
    exit()

# 확인
print("\n삭제할 파일 목록 (처음 20개):")
for i, f in enumerate(to_delete_images[:20]):
    print(f"  이미지: {f.name}")
if len(to_delete_images) > 20:
    print(f"  ... 외 {len(to_delete_images) - 20}개")

for i, f in enumerate(to_delete_labels[:20]):
    print(f"  라벨: {f.name}")
if len(to_delete_labels) > 20:
    print(f"  ... 외 {len(to_delete_labels) - 20}개")

print("\n삭제를 시작합니다...")

# 삭제 실행
deleted_images = 0
deleted_labels = 0

for img_file in to_delete_images:
    try:
        os.remove(img_file)
        deleted_images += 1
    except Exception as e:
        print(f"  ❌ 삭제 실패: {img_file.name} - {e}")

for label_file in to_delete_labels:
    try:
        os.remove(label_file)
        deleted_labels += 1
    except Exception as e:
        print(f"  ❌ 삭제 실패: {label_file.name} - {e}")

print(f"\n삭제 완료:")
print(f"  이미지: {deleted_images}개")
print(f"  라벨: {deleted_labels}개")

# 최종 통계
remaining_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
remaining_labels = len(list(labels_dir.glob('*.txt')))

print(f"\n남은 파일:")
print(f"  이미지: {remaining_images}개")
print(f"  라벨: {remaining_labels}개")
print("=" * 70)
