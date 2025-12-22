"""
Data_polygon 폴더의 이미지에 세그멘테이션 마스크 시각화
(past_roboflow_visualized_polygon 스타일)
"""
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(r'c:\Users\24457\OneDrive\바탕 화면\MCA')
POLYGON_DIR = BASE_DIR / 'Data_polygon'

# 클래스 정보
class_names = {0: 'combined', 1: 'empty', 2: 'fully'}
class_colors = {
    0: (255, 0, 0),    # combined: 파랑 (BGR)
    1: (0, 0, 255),    # empty: 빨강
    2: (0, 255, 0)     # fully: 초록
}

print("="*70)
print("세그멘테이션 마스크 시각화")
print("="*70)

# Data 폴더 사용 (원본 세그멘테이션 라벨)
DATA_DIR = BASE_DIR / 'Data'

total_processed = 0

for split in ['train', 'valid', 'test']:
    polygon_split = POLYGON_DIR / split
    data_split = DATA_DIR / split

    if not polygon_split.exists():
        continue

    images_dir = polygon_split / 'images'
    labels_dir = polygon_split / 'labels'

    # Data 폴더의 라벨 사용
    data_labels_dir = data_split / 'labels'

    if not images_dir.exists():
        continue

    print(f"\n[{split}] 처리 중...")

    image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    processed = 0

    for img_file in image_files:
        # 이미지 로드
        img = cv2.imread(str(img_file))

        if img is None:
            continue

        # 대응하는 라벨 파일 찾기 (Data 폴더에서)
        label_name = img_file.stem + '.txt'

        # Data/train, Data/valid에서 찾기
        label_file = None
        if data_labels_dir.exists():
            # 이름 매칭 시도
            possible_labels = list(data_labels_dir.glob(f"{split}_*.txt"))

            # 번호 추출해서 매칭
            try:
                img_num = int(img_file.stem.split('_')[1])
                for lbl in possible_labels:
                    lbl_num = int(lbl.stem.split('_')[1])
                    if lbl_num == img_num:
                        label_file = lbl
                        break
            except:
                pass

        # test는 Data_polygon 자체 라벨 사용
        if label_file is None and labels_dir.exists():
            label_file = labels_dir / label_name

        if label_file is None or not label_file.exists():
            continue

        # 라벨 읽기
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                # 세그멘테이션 형식인지 확인 (좌표 5개 이상)
                if len(coords) <= 4:
                    # bbox 형식이면 건너뛰기
                    continue

                # 절대 좌표로 변환
                h, w = img.shape[:2]
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)

                # 폴리곤 그리기
                color = class_colors.get(class_id, (128, 128, 128))
                cv2.polylines(img, [points], isClosed=True, color=color, thickness=3)

                # 바운딩 박스 계산 (라벨 표시용)
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # 클래스명 표시
                class_name = class_names.get(class_id, f'class_{class_id}')
                label_text = class_name

                # 텍스트 배경
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, (x_min, y_min - 30), (x_min + text_w + 10, y_min), color, -1)

                # 텍스트
                cv2.putText(img, label_text, (x_min + 5, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 덮어쓰기
        cv2.imwrite(str(img_file), img)
        processed += 1

    print(f"  → {processed}개 이미지 처리 완료")
    total_processed += processed

    # 라벨 폴더 삭제
    if labels_dir.exists():
        import shutil
        shutil.rmtree(labels_dir)
        print(f"  → labels 폴더 삭제 완료")

print("\n" + "="*70)
print(f"전체 {total_processed}개 이미지 시각화 완료!")
print("라벨 파일 전부 삭제 완료!")
print("="*70)
