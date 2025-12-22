"""
최종 데이터셋 확인
"""
from pathlib import Path

print("="*60)
print("Final Dataset Check")
print("="*60)

total_train_images = 0
total_train_labels = 0
total_valid_images = 0
total_valid_labels = 0

for split in ['train', 'valid']:
    labels_dir = Path(f'roboflow/{split}/labels')
    images_dir = Path(f'roboflow/{split}/images')

    images = list(images_dir.glob('*.jpg'))
    labels = list(labels_dir.glob('*.txt'))

    print(f"\n{split.upper()}:")
    print(f"  Images: {len(images)}")
    print(f"  Labels: {len(labels)}")
    print(f"  Match:  {'YES' if len(images) == len(labels) else 'NO'}")

    if split == 'train':
        total_train_images = len(images)
        total_train_labels = len(labels)
    else:
        total_valid_images = len(images)
        total_valid_labels = len(labels)

    # 매칭 확인
    image_stems = {img.stem for img in images}
    label_stems = {lbl.stem for lbl in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    if missing_labels:
        print(f"  Missing labels for: {missing_labels}")
    if missing_images:
        print(f"  Missing images for: {missing_images}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total images: {total_train_images + total_valid_images}")
print(f"Total labels: {total_train_labels + total_valid_labels}")
print(f"Train: {total_train_images} images, {total_train_labels} labels")
print(f"Valid: {total_valid_images} images, {total_valid_labels} labels")

total = total_train_images + total_valid_images
print(f"\nTrain ratio: {total_train_images/total*100:.1f}%")
print(f"Valid ratio: {total_valid_images/total*100:.1f}%")
print("="*60)
