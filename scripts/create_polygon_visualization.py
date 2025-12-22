import cv2
import numpy as np
from pathlib import Path
import shutil

DATA_DIR = Path('Data')
OUTPUT_DIR = Path('Data_polygon')

# Class names and colors
CLASS_NAMES = {0: 'combined', 1: 'empty', 2: 'fully'}
CLASS_COLORS = {
    0: (255, 0, 0),      # Blue for combined
    1: (0, 255, 0),      # Green for empty
    2: (0, 0, 255)       # Red for fully
}

# Create output directory structure
for split in ['train', 'valid']:
    output_split_dir = OUTPUT_DIR / split / 'images'
    output_split_dir.mkdir(parents=True, exist_ok=True)

print("Creating Data_polygon with visualized segmentation masks...\n")

total_processed = 0

for split in ['train', 'valid']:
    images_dir = DATA_DIR / split / 'images'
    labels_dir = DATA_DIR / split / 'labels'
    output_dir = OUTPUT_DIR / split / 'images'

    if not images_dir.exists():
        continue

    image_files = list(images_dir.glob('*.jpg'))
    processed = 0

    for img_file in image_files:
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not read {img_file}")
            continue

        h, w = img.shape[:2]

        # Read corresponding label file
        label_file = labels_dir / (img_file.stem + '.txt')

        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                values = list(map(float, line.strip().split()))
                class_id = int(values[0])

                # Convert normalized coordinates to pixel coordinates
                points = []
                for i in range(1, len(values), 2):
                    x = int(values[i] * w)
                    y = int(values[i+1] * h)
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)

                # Draw polygon
                cv2.polylines(img, [points], True, CLASS_COLORS[class_id], 2)

                # Add class label
                centroid_x = int(np.mean(points[:, 0]))
                centroid_y = int(np.mean(points[:, 1]))

                label_text = CLASS_NAMES[class_id]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                # Get text size for background
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                # Draw background rectangle
                cv2.rectangle(img,
                            (centroid_x - 5, centroid_y - text_h - 5),
                            (centroid_x + text_w + 5, centroid_y + 5),
                            CLASS_COLORS[class_id], -1)

                # Draw text
                cv2.putText(img, label_text, (centroid_x, centroid_y),
                           font, font_scale, (255, 255, 255), thickness)

        # Save visualized image
        output_path = output_dir / img_file.name
        cv2.imwrite(str(output_path), img)
        processed += 1

    print(f"{split}: {processed} images processed")
    total_processed += processed

print(f"\nTotal: {total_processed} images with visualized segmentation masks")
print(f"Output saved to: {OUTPUT_DIR}")
