"""
Roboflow 폴리곤 세그멘테이션 시각화
(YOLO OBB/Segmentation 형식 지원)
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 클래스 정보
class_names = {0: 'combined', 1: 'empty', 2: 'fully'}
class_colors = {
    0: (255, 0, 0),    # combined: 파랑
    1: (0, 0, 255),    # empty: 빨강
    2: (0, 255, 0)     # fully: 초록
}

def visualize_folder(images_dir, labels_dir, output_dir):
    """폴더 내 모든 이미지에 폴리곤/바운딩 박스 그리기"""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob('*.jpg'))
    print(f"\nProcessing {len(image_files)} images from {images_dir.parent.name}/{images_dir.name}...")

    success_count = 0

    for img_path in tqdm(image_files, desc=f"  {images_dir.parent.name}"):
        # 이미지 읽기
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # 라벨 파일 읽기
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        # 각 객체 처리
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls = int(float(parts[0]))
                color = class_colors[cls]

                # 바운딩 박스 형식 (5개 필드)
                if len(parts) == 5:
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    box_w = float(parts[3]) * w
                    box_h = float(parts[4]) * h

                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)

                    # 바운딩 박스 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                    # 라벨
                    label = class_names[cls]
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img, (x1, y1 - 30), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(img, label, (x1 + 5, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # 폴리곤 형식 (홀수개 필드: class + (x,y) 쌍들)
                else:
                    # 좌표 추출
                    coords = []
                    for i in range(1, len(parts), 2):
                        if i+1 < len(parts):
                            x = float(parts[i]) * w
                            y = float(parts[i+1]) * h
                            coords.append([int(x), int(y)])

                    if len(coords) >= 3:  # 최소 3개 점 필요
                        # 폴리곤 그리기
                        pts = np.array(coords, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], True, color, 3)

                        # 폴리곤의 바운딩 박스 계산
                        x_coords = [c[0] for c in coords]
                        y_coords = [c[1] for c in coords]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)

                        # 라벨 (바운딩 박스 좌상단에)
                        label = class_names[cls]
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(img, (x1, y1 - 30), (x1 + tw + 10, y1), color, -1)
                        cv2.putText(img, label, (x1 + 5, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 저장
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), img)
        success_count += 1

    return success_count

# 메인
print("="*60)
print("Roboflow Polygon Segmentation Visualization")
print("="*60)

roboflow_root = Path('roboflow')
output_root = Path('roboflow_visualized_polygon')

# Train
train_count = visualize_folder(
    roboflow_root / 'train' / 'images',
    roboflow_root / 'train' / 'labels',
    output_root / 'train'
)

# Valid
valid_count = visualize_folder(
    roboflow_root / 'valid' / 'images',
    roboflow_root / 'valid' / 'labels',
    output_root / 'valid'
)

# Test
test_count = visualize_folder(
    roboflow_root / 'test' / 'images',
    roboflow_root / 'test' / 'labels',
    output_root / 'test'
)

# 결과
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Train: {train_count} images")
print(f"Valid: {valid_count} images")
print(f"Test:  {test_count} images")
print(f"Total: {train_count + valid_count + test_count} images")
print(f"\nOutput folder: {output_root}")
print("="*60)
