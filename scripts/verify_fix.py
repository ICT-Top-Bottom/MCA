"""
수정 결과 확인
"""
from pathlib import Path

def count_annotations(label_file):
    boxes = 0
    segments = 0
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if len(parts) == 5:
                boxes += 1
            else:
                segments += 1
    return boxes, segments

print("="*60)
print("Verification After Fix")
print("="*60)

for split in ['train', 'valid']:
    labels_dir = Path(f'roboflow/{split}/labels')

    boxes = 0
    segments = 0
    files = 0

    for label_file in labels_dir.glob('*.txt'):
        b, s = count_annotations(label_file)
        boxes += b
        segments += s
        files += 1

    print(f"\n{split.upper()}:")
    print(f"  Files:    {files}")
    print(f"  Boxes:    {boxes}")
    print(f"  Segments: {segments}")
    print(f"  Match:    {'YES' if boxes == segments else 'NO'}")

print("\n" + "="*60)
