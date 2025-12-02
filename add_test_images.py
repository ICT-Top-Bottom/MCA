"""
TEST/labeledImages에 남은 이미지들을 images와 labels에 추가
"""
from pathlib import Path
import shutil

base_dir = Path(__file__).parent
test_labeled_dir = base_dir / 'TEST' / 'labeledImages'
test_images_dir = base_dir / 'TEST' / 'empty_exif'
test_labels_dir = base_dir / 'TEST' / 'empty_exif_label'

main_images_dir = base_dir / 'images'
main_labels_dir = base_dir / 'labels'

print("=" * 70)
print("TEST 이미지를 메인 폴더에 추가")
print("=" * 70)

# labeledImages에 남아있는 파일들 (삭제 안 된 것들)
remaining_files = set()
for img_file in test_labeled_dir.glob('*.jpg'):
    remaining_files.add(img_file.stem)

print(f"TEST/labeledImages에 남은 파일: {len(remaining_files)}개\n")

if len(remaining_files) == 0:
    print("추가할 파일이 없습니다.")
    print("=" * 70)
    exit()

# 추가할 파일 수집
to_add_images = []
to_add_labels = []

for stem in remaining_files:
    # 이미지 파일 찾기
    img_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = test_images_dir / f"{stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if img_path:
        to_add_images.append(img_path)

    # 라벨 파일 찾기
    label_path = test_labels_dir / f"{stem}.txt"
    if label_path.exists():
        to_add_labels.append(label_path)

print(f"추가할 이미지: {len(to_add_images)}개")
print(f"추가할 라벨: {len(to_add_labels)}개")

# 파일 복사
added_images = 0
added_labels = 0

print("\n복사 중...")

for img_file in to_add_images:
    try:
        dest = main_images_dir / img_file.name
        shutil.copy2(img_file, dest)
        added_images += 1
    except Exception as e:
        print(f"  ❌ 이미지 복사 실패: {img_file.name} - {e}")

for label_file in to_add_labels:
    try:
        dest = main_labels_dir / label_file.name
        shutil.copy2(label_file, dest)
        added_labels += 1
    except Exception as e:
        print(f"  ❌ 라벨 복사 실패: {label_file.name} - {e}")

print(f"\n추가 완료:")
print(f"  이미지: {added_images}개")
print(f"  라벨: {added_labels}개")

# 최종 통계
print("\n" + "=" * 70)
print("최종 데이터셋 통계")
print("=" * 70)

# 클래스별 개수 계산
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
total_objects = 0
for class_id in sorted(class_counts.keys()):
    count = class_counts[class_id]
    total_objects += count
    percentage = (count / total_objects * 100) if total_objects > 0 else 0
    print(f"  {class_names[class_id]}: {count}개")

print(f"\n총 객체 수: {total_objects}개")

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

print("=" * 70)
