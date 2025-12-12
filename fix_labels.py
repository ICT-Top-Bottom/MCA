"""
문제가 있는 라벨 파일과 해당 이미지 삭제
"""
from pathlib import Path
import shutil

# 문제 파일 리스트
problem_files = ['0001.txt', '0036.txt']

print("=" * 60)
print("Removing problematic label files and images")
print("=" * 60)

removed_count = 0

for split in ['train']:
    labels_dir = Path(f'roboflow/{split}/labels')
    images_dir = Path(f'roboflow/{split}/images')

    print(f"\nProcessing {split}:")

    for label_file in problem_files:
        label_path = labels_dir / label_file
        image_name = label_file.replace('.txt', '.jpg')
        image_path = images_dir / image_name

        # 라벨 삭제
        if label_path.exists():
            label_path.unlink()
            print(f"  ✓ Removed label: {label_file}")
            removed_count += 1

        # 이미지 삭제
        if image_path.exists():
            image_path.unlink()
            print(f"  ✓ Removed image: {image_name}")

print("\n" + "=" * 60)
print(f"Removed {removed_count} problematic files")
print("=" * 60)

# 검증
print("\nVerifying fix...")
from diagnose_labels import diagnose_dataset
problems = diagnose_dataset('roboflow')

if len(problems) == 0:
    print("\n✅ All issues resolved!")
else:
    print(f"\n⚠️  Still {len(problems)} problems remaining")
