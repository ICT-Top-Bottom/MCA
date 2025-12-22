"""
빈 annotation 또는 비정상 annotation 찾기
"""
from pathlib import Path

print("="*60)
print("Checking for empty or invalid annotations")
print("="*60)

empty_files = []
invalid_files = []

for split in ['train', 'valid']:
    labels_dir = Path(f'roboflow/{split}/labels')
    images_dir = Path(f'roboflow/{split}/images')

    print(f"\n{split.upper()}:")

    for label_file in sorted(labels_dir.glob('*.txt')):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            empty_files.append((split, label_file.name))
            print(f"  EMPTY: {label_file.name}")
            continue

        # 각 라인 검증
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:  # class + 최소 2개 좌표 (x,y)
                invalid_files.append((split, label_file.name, i+1, len(parts)))
                print(f"  INVALID: {label_file.name} line {i+1} (only {len(parts)} values)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Empty files:   {len(empty_files)}")
print(f"Invalid files: {len(invalid_files)}")

if empty_files:
    print("\nRemoving empty annotation files and images...")
    for split, filename in empty_files:
        label_path = Path(f'roboflow/{split}/labels/{filename}')
        image_name = filename.replace('.txt', '.jpg')
        image_path = Path(f'roboflow/{split}/images/{image_name}')

        if label_path.exists():
            label_path.unlink()
            print(f"  Removed: {split}/labels/{filename}")

        if image_path.exists():
            image_path.unlink()
            print(f"  Removed: {split}/images/{image_name}")

print("\nDone!")
