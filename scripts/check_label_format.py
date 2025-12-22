"""
라벨 형식 확인 스크립트
"""
from pathlib import Path
from collections import Counter

labels_dir = Path('roboflow/train/labels')

field_counts = Counter()
sample_lines = {}

for label_file in list(labels_dir.glob('*.txt'))[:50]:  # 50개만 확인
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            field_count = len(parts)
            field_counts[field_count] += 1

            if field_count not in sample_lines:
                sample_lines[field_count] = (label_file.name, line.strip())

print("Field Count Distribution:")
for count, freq in sorted(field_counts.items()):
    print(f"  {count} fields: {freq} lines")

print("\nSample lines:")
for count in sorted(sample_lines.keys()):
    filename, line = sample_lines[count]
    print(f"\n{count} fields from {filename}:")
    print(f"  {line[:100]}...")
