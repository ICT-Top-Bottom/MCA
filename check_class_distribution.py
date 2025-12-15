"""
클래스별 데이터 분포 확인
"""
from pathlib import Path
from collections import Counter

BASE_DIR = Path(r'c:\Users\24457\OneDrive\바탕 화면\MCA')

print("="*70)
print("클래스별 데이터 분포")
print("="*70)

class_names = {0: 'combined', 1: 'empty', 2: 'fully'}

# Data와 Data_polygon 모두 체크
for dataset_name in ['Data', 'Data_polygon']:
    dataset_dir = BASE_DIR / dataset_name

    if not dataset_dir.exists():
        continue

    print(f"\n{'='*70}")
    print(f"{dataset_name}/")
    print("="*70)

    dataset_total = Counter()

    for split in ['train', 'valid', 'test']:
        split_dir = dataset_dir / split

        if not split_dir.exists():
            continue

        labels_dir = split_dir / 'labels'
        images_dir = split_dir / 'images'

        if not labels_dir.exists():
            continue

        # 클래스별 카운터
        class_counter = Counter()
        total_objects = 0
        image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png'))) if images_dir.exists() else 0

        # 라벨 파일 읽기
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counter[class_id] += 1
                        dataset_total[class_id] += 1
                        total_objects += 1

        print(f"\n[{split}]")
        print(f"이미지: {image_count}개")
        print(f"전체 객체: {total_objects}개")

        if total_objects > 0:
            for class_id in sorted(class_counter.keys()):
                count = class_counter[class_id]
                class_name = class_names.get(class_id, f'class_{class_id}')
                percentage = (count / total_objects * 100)
                print(f"  {class_name}: {count}개 ({percentage:.1f}%)")

    # 데이터셋 전체 통계
    total_all = sum(dataset_total.values())
    if total_all > 0:
        print(f"\n{'='*70}")
        print(f"{dataset_name} 전체 통계")
        print("="*70)
        print(f"전체 객체 수: {total_all}개\n")
        for class_id in sorted(dataset_total.keys()):
            count = dataset_total[class_id]
            class_name = class_names.get(class_id, f'class_{class_id}')
            percentage = (count / total_all * 100)
            print(f"  {class_name}: {count}개 ({percentage:.1f}%)")

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
