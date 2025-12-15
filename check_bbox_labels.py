from pathlib import Path

DATA_DIR = Path('Data')

def is_bbox_format(label_file):
    """Check if label file is bbox format (5 values) instead of segmentation format"""
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            values = line.strip().split()
            if len(values) == 5:  # bbox format: class x_center y_center width height
                return True
        return False
    except:
        return False

bbox_files = []

for split in ['train', 'valid']:
    labels_dir = DATA_DIR / split / 'labels'
    if not labels_dir.exists():
        continue

    for label_file in labels_dir.glob('*.txt'):
        if is_bbox_format(label_file):
            img_name = label_file.stem + '.jpg'
            bbox_files.append((split, img_name, label_file))

if bbox_files:
    print(f"Found {len(bbox_files)} bbox format files:")
    for split, img_name, label_file in bbox_files:
        print(f"  {split}/{img_name}")
else:
    print("No bbox format files found - all labels are segmentation format")

print(f"\nTotal labels checked:")
for split in ['train', 'valid']:
    labels_dir = DATA_DIR / split / 'labels'
    if labels_dir.exists():
        count = len(list(labels_dir.glob('*.txt')))
        print(f"  {split}: {count}")
