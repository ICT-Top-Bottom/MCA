from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('Data')
CLASS_NAMES = {0: 'combined', 1: 'empty', 2: 'fully'}

def analyze_split(split_name):
    labels_dir = DATA_DIR / split_name / 'labels'
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
print("클래스별 분포")
print("=" * 60)

for split in ['train', 'valid']:
    counts = analyze_split(split)
    if counts is None:
        continue

    total = sum(counts.values())
    print(f"\n[{split.upper()}] - 총 {total}개 객체")
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
print(f"\n[전체] - 총 {total_all}개 객체")
for class_id in sorted(all_counts.keys()):
    count = all_counts[class_id]
    pct = count/total_all*100
    print(f"  {CLASS_NAMES[class_id]}: {count}개 ({pct:.1f}%)")

print("\n클래스 비율:")
combined = all_counts[0]
empty = all_counts[1]
fully = all_counts[2]
print(f"  combined : empty : fully = {combined} : {empty} : {fully}")
print(f"  = 1 : {empty/combined:.2f} : {fully/combined:.2f}")
print("=" * 60)
