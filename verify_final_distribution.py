from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('Data')

CLASS_NAMES = {0: 'combined', 1: 'empty', 2: 'fully'}

def analyze_split(split_name):
    labels_dir = DATA_DIR / split_name / 'labels'
    images_dir = DATA_DIR / split_name / 'images'

    if not labels_dir.exists():
        return None

    class_counts = defaultdict(int)
    total_objects = 0
    total_images = len(list(images_dir.glob('*.jpg')))

    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            values = line.strip().split()
            class_id = int(values[0])
            class_counts[class_id] += 1
            total_objects += 1

    return {
        'images': total_images,
        'objects': total_objects,
        'classes': class_counts
    }

print("=" * 60)
print("FINAL DATA DISTRIBUTION")
print("=" * 60)

total_images_all = 0
total_objects_all = 0
class_totals = defaultdict(int)

for split in ['train', 'valid']:
    result = analyze_split(split)
    if result is None:
        continue

    print(f"\n[{split.upper()}]")
    print(f"Images: {result['images']}")
    print(f"Total objects: {result['objects']}")

    for class_id in sorted(result['classes'].keys()):
        count = result['classes'][class_id]
        percentage = (count / result['objects'] * 100) if result['objects'] > 0 else 0
        print(f"  {CLASS_NAMES[class_id]}: {count} ({percentage:.1f}%)")

    total_images_all += result['images']
    total_objects_all += result['objects']
    for class_id, count in result['classes'].items():
        class_totals[class_id] += count

print(f"\n{'=' * 60}")
print(f"[OVERALL]")
print(f"Total images: {total_images_all}")
print(f"Total objects: {total_objects_all}")

for class_id in sorted(class_totals.keys()):
    count = class_totals[class_id]
    percentage = (count / total_objects_all * 100) if total_objects_all > 0 else 0
    print(f"  {CLASS_NAMES[class_id]}: {count} ({percentage:.1f}%)")

print("=" * 60)
