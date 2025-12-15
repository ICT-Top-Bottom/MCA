from pathlib import Path
import shutil
import os

DATA_DIR = Path('Data')
POLYGON_DIR = Path('Data_polygon')

# Step 1: Delete files from Data/valid
deleted_files = ['valid_0002.jpg', 'valid_0007.jpg', 'valid_0014.jpg']

print("Step 1: Data/valid에서 삭제된 파일 제거")
for filename in deleted_files:
    img_path = DATA_DIR / 'valid' / 'images' / filename
    label_path = DATA_DIR / 'valid' / 'labels' / (filename.replace('.jpg', '.txt'))

    if img_path.exists():
        os.remove(img_path)
        print(f"  이미지 삭제: {filename}")

    if label_path.exists():
        os.remove(label_path)
        print(f"  라벨 삭제: {filename.replace('.jpg', '.txt')}")

# Step 2: Find combined class images in train
print("\nStep 2: Train에서 combined 클래스 이미지 찾기")
train_labels_dir = DATA_DIR / 'train' / 'labels'
combined_files = []

for label_file in train_labels_dir.glob('*.txt'):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    # Check if any line has class 0 (combined)
    has_combined = any(line.strip().startswith('0 ') for line in lines)

    if has_combined:
        combined_files.append(label_file.stem + '.jpg')

print(f"  Combined 클래스 이미지: {len(combined_files)}개 발견")

# Select first 3 combined files
files_to_move = combined_files[:3]
print(f"  이동할 파일 선택: {files_to_move}")

# Step 3: Move files from train to valid
print("\nStep 3: Train → Valid 이동")
for filename in files_to_move:
    # Move image
    src_img = DATA_DIR / 'train' / 'images' / filename
    dst_img = DATA_DIR / 'valid' / 'images' / filename
    shutil.move(str(src_img), str(dst_img))
    print(f"  이미지 이동: {filename}")

    # Move label
    label_name = filename.replace('.jpg', '.txt')
    src_label = DATA_DIR / 'train' / 'labels' / label_name
    dst_label = DATA_DIR / 'valid' / 'labels' / label_name
    shutil.move(str(src_label), str(dst_label))
    print(f"  라벨 이동: {label_name}")

    # Create visualization for Data_polygon/valid
    import cv2
    import numpy as np

    img = cv2.imread(str(dst_img))
    if img is not None:
        h, w = img.shape[:2]

        CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
        CLASS_NAMES = {0: 'combined', 1: 'empty', 2: 'fully'}

        with open(dst_label, 'r') as f:
            lines = f.readlines()

        for line in lines:
            values = list(map(float, line.strip().split()))
            class_id = int(values[0])

            points = []
            for i in range(1, len(values), 2):
                x = int(values[i] * w)
                y = int(values[i+1] * h)
                points.append([x, y])

            points = np.array(points, dtype=np.int32)
            cv2.polylines(img, [points], True, CLASS_COLORS[class_id], 2)

            centroid_x = int(np.mean(points[:, 0]))
            centroid_y = int(np.mean(points[:, 1]))

            label_text = CLASS_NAMES[class_id]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            cv2.rectangle(img, (centroid_x - 5, centroid_y - text_h - 5),
                        (centroid_x + text_w + 5, centroid_y + 5),
                        CLASS_COLORS[class_id], -1)
            cv2.putText(img, label_text, (centroid_x, centroid_y),
                       font, font_scale, (255, 255, 255), thickness)

        # Save to Data_polygon/valid
        polygon_output = POLYGON_DIR / 'valid' / 'images' / filename
        cv2.imwrite(str(polygon_output), img)
        print(f"  시각화 생성: Data_polygon/valid/images/{filename}")

print("\n완료!")
print(f"Data/valid 이미지: {len(list((DATA_DIR / 'valid' / 'images').glob('*.jpg')))}개")
print(f"Data/train 이미지: {len(list((DATA_DIR / 'train' / 'images').glob('*.jpg')))}개")
print(f"Data_polygon/valid 이미지: {len(list((POLYGON_DIR / 'valid' / 'images').glob('*.jpg')))}개")
