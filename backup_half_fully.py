"""
fully 이미지를 2장당 1장씩 backupFully 폴더로 이동
"""
from pathlib import Path
import shutil

base_dir = Path(__file__).parent
images_dir = base_dir / 'images'
labels_dir = base_dir / 'labels'

backup_dir = base_dir / 'backupFully'
backup_images_dir = backup_dir / 'images'
backup_labels_dir = backup_dir / 'labels'

backup_images_dir.mkdir(parents=True, exist_ok=True)
backup_labels_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Fully 이미지 절반 백업")
print("=" * 70)

# fully 파일들 찾기
fully_files = sorted([f.stem for f in images_dir.glob('fully_*')])

print(f"\n전체 fully 파일: {len(fully_files)}개")

# 2장당 1장씩 선택 (홀수 인덱스만)
to_backup = []
for i, stem in enumerate(fully_files):
    if i % 2 == 1:  # 홀수 인덱스 (0, 2, 4... 는 남기고, 1, 3, 5... 는 백업)
        to_backup.append(stem)

print(f"백업할 파일: {len(to_backup)}개")
print(f"남을 파일: {len(fully_files) - len(to_backup)}개")

# 백업 실행
backed_up_images = 0
backed_up_labels = 0

print("\n백업 중...")

for stem in to_backup:
    # 이미지 백업
    for ext in ['.jpg', '.png', '.jpeg']:
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            shutil.move(str(img_path), str(backup_images_dir / img_path.name))
            backed_up_images += 1
            break

    # 라벨 백업
    label_path = labels_dir / f"{stem}.txt"
    if label_path.exists():
        shutil.move(str(label_path), str(backup_labels_dir / label_path.name))
        backed_up_labels += 1

print(f"\n백업 완료:")
print(f"  이미지: {backed_up_images}개")
print(f"  라벨: {backed_up_labels}개")

# 최종 통계
print("\n" + "=" * 70)
print("최종 통계")
print("=" * 70)

from collections import defaultdict

class_counts = defaultdict(int)
class_names = {0: 'fully', 1: 'empty', 2: 'combined'}

for label_file in labels_dir.glob('*.txt'):
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                class_counts[class_id] += 1

total_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
total_labels = len(list(labels_dir.glob('*.txt')))

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

print("\n" + "=" * 70)
print(f"백업 위치: {backup_dir}/")
print("=" * 70)
