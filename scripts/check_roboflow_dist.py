from pathlib import Path
from collections import defaultdict

ROBOFLOW_DIR = Path('roboflow')
CLASS_NAMES = {0: 'combined', 1: 'empty', 2: 'fully'}

def analyze_split(split_name):
    labels_dir = ROBOFLOW_DIR / split_name / 'labels'
    if not labels_dir.exists():
        return None

    class_counts = defaultdict(int)
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f.readlines():
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1

    return class_counts

print("=" * 60)
print("Roboflow 데이터셋 클래스별 분포")
print("=" * 60)

for split in ['train', 'valid']:
    counts = analyze_split(split)
    if counts is None:
        continue

    total = sum(counts.values())
    images = len(list((ROBOFLOW_DIR / split / 'images').glob('*.jpg')))
    print(f"\n[{split.upper()}] - {images}장 이미지, 총 {total}개 객체")
    for class_id in sorted(counts.keys()):
        count = counts[class_id]
        pct = count/total*100
        print(f"  {CLASS_NAMES[class_id]}: {count}개 ({pct:.1f}%)")

# Overall
all_counts = defaultdict(int)
for split in ['train', 'valid']:
    counts = analyze_split(split)
    if counts:
        for class_id, count in counts.items():
            all_counts[class_id] += count

total_all = sum(all_counts.values())
total_images = len(list((ROBOFLOW_DIR / 'train' / 'images').glob('*.jpg'))) + \
               len(list((ROBOFLOW_DIR / 'valid' / 'images').glob('*.jpg')))

print(f"\n[전체] - {total_images}장 이미지, 총 {total_all}개 객체")
for class_id in sorted(all_counts.keys()):
    count = all_counts[class_id]
    pct = count/total_all*100
    print(f"  {CLASS_NAMES[class_id]}: {count}개 ({pct:.1f}%)")

print("=" * 60)
