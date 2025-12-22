"""
Data_polygon 폴더의 세그멘테이션 라벨을 바운딩박스 형식으로 변환
"""
from pathlib import Path

POLYGON_DIR = Path(r'c:\Users\24457\OneDrive\바탕 화면\MCA\Data_polygon')

print("="*70)
print("세그멘테이션 → 바운딩박스 변환")
print("="*70)

def seg_to_bbox(seg_coords):
    """
    세그멘테이션 좌표 리스트 → 바운딩박스 (x_center, y_center, width, height)
    """
    # x, y 좌표 분리
    x_coords = seg_coords[::2]  # 짝수 인덱스
    y_coords = seg_coords[1::2]  # 홀수 인덱스

    # 최소/최대값 계산
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # 중심점과 크기 계산
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height

total_converted = 0

for split in ['train', 'valid', 'test']:
    labels_dir = POLYGON_DIR / split / 'labels'

    if not labels_dir.exists():
        continue

    print(f"\n[{split}] 변환 중...")
    converted_count = 0

    for label_file in labels_dir.glob('*.txt'):
        new_lines = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 5:
                    # 빈 줄이나 잘못된 형식
                    continue

                class_id = parts[0]
                coords = [float(x) for x in parts[1:]]

                # 세그멘테이션 형식 (5개 이상 좌표)
                if len(coords) > 4:
                    x_center, y_center, width, height = seg_to_bbox(coords)
                    new_line = f"{class_id} {x_center} {y_center} {width} {height}"
                    new_lines.append(new_line)
                else:
                    # 이미 bbox 형식
                    new_lines.append(line.strip())

        # 파일 덮어쓰기
        with open(label_file, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')

        converted_count += 1

    print(f"  → {converted_count}개 파일 변환 완료")
    total_converted += converted_count

print("\n" + "="*70)
print(f"전체 {total_converted}개 라벨 파일 변환 완료!")
print("="*70)

# 변환 확인
print("\n변환 결과 확인:")
sample_file = list((POLYGON_DIR / 'train' / 'labels').glob('*.txt'))[0]
print(f"\n샘플: {sample_file.name}")
with open(sample_file, 'r') as f:
    for i, line in enumerate(f):
        if i < 3:
            print(f"  {line.strip()}")
        else:
            break
