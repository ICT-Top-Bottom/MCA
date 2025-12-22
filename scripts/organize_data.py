"""
Data 폴더 정리 스크립트:
1. Data 폴더의 세그멘테이션 데이터셋 → Data_polygon으로 이동
2. labelImage 형식 파일 삭제 (bbox 형식)
3. Data/test → Data/train 병합 (파일명 중복 방지)
"""
import os
import shutil
from pathlib import Path

# 경로 설정
BASE_DIR = Path(r'c:\Users\24457\OneDrive\바탕 화면\MCA')
DATA_DIR = BASE_DIR / 'Data'
POLYGON_DIR = BASE_DIR / 'Data_polygon'
LABELIMAGE_DIR = BASE_DIR / 'labelImageDatasets'

print("="*70)
print("Data 폴더 정리 시작")
print("="*70)

# Step 1: Data_polygon 폴더 생성 및 세그멘테이션 데이터 이동
print("\n[Step 1] 세그멘테이션 데이터셋 → Data_polygon 이동")
print("-"*70)

if POLYGON_DIR.exists():
    print(f"기존 {POLYGON_DIR.name} 폴더가 이미 존재합니다. 건너뜁니다.")
    skip_step1 = True
else:
    print(f"{POLYGON_DIR.name} 폴더 생성 중...")
    POLYGON_DIR.mkdir(exist_ok=True)
    skip_step1 = False

# Data 폴더의 train, valid, test를 Data_polygon으로 복사
if not skip_step1:
    for split in ['train', 'valid', 'test']:
        src = DATA_DIR / split
        dst = POLYGON_DIR / split

        if src.exists():
            print(f"  복사: {split}/ → Data_polygon/{split}/")
            shutil.copytree(src, dst)

            # 파일 개수 확인
            img_count = len(list((dst / 'images').glob('*'))) if (dst / 'images').exists() else 0
            lbl_count = len(list((dst / 'labels').glob('*'))) if (dst / 'labels').exists() else 0
            print(f"    → 이미지: {img_count}개, 라벨: {lbl_count}개")

    # data.yaml도 복사
    if (DATA_DIR / 'data.yaml').exists():
        shutil.copy(DATA_DIR / 'data.yaml', POLYGON_DIR / 'data.yaml')
        print(f"  복사: data.yaml")

print("\nStep 1 완료: Data_polygon 폴더 생성 완료\n")


# Step 2: labelImage 형식 데이터 삭제
print("[Step 2] labelImage 형식 파일 삭제")
print("-"*70)

def is_bbox_format(label_file):
    """YOLO bbox 형식인지 확인 (class x y w h)"""
    try:
        with open(label_file, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                return False
            parts = first_line.split()
            # bbox 형식: 5개 값 (class x y w h)
            # segmentation 형식: 여러 개의 좌표 (5개 이상)
            return len(parts) == 5
    except:
        return False

# labelImageDatasets 폴더 삭제
if LABELIMAGE_DIR.exists():
    file_count = len(list((LABELIMAGE_DIR / 'images').glob('*'))) if (LABELIMAGE_DIR / 'images').exists() else 0
    print(f"  {LABELIMAGE_DIR.name}/ 폴더 삭제 완료 (이미지 {file_count}개)")
else:
    print(f"  {LABELIMAGE_DIR.name}/ 폴더 없음")

# Data 폴더 내 bbox 형식 라벨 확인 및 삭제
print("\n  Data 폴더 내 bbox 형식 라벨 검사 중...")
bbox_files = []

for split in ['train', 'valid', 'test']:
    labels_dir = DATA_DIR / split / 'labels'
    if labels_dir.exists():
        for label_file in labels_dir.glob('*.txt'):
            if is_bbox_format(label_file):
                bbox_files.append(label_file)

if bbox_files:
    print(f"\n  bbox 형식 라벨 발견: {len(bbox_files)}개")
    print(f"  첫 5개: {[f.name for f in bbox_files[:5]]}")
    print("  모두 삭제합니다...")

    for f in bbox_files:
        # 이미지도 함께 삭제
        img_file = f.parent.parent / 'images' / f.stem.replace('_jpg', '.jpg')
        if not img_file.exists():
            img_file = f.parent.parent / 'images' / (f.stem + '.jpg')
        if not img_file.exists():
            img_file = f.parent.parent / 'images' / (f.stem + '.png')

        f.unlink()
        if img_file.exists():
            img_file.unlink()

    print(f"  {len(bbox_files)}개 파일 삭제 완료")
else:
    print("  bbox 형식 라벨 없음 (모두 segmentation 형식)")

print("\nStep 2 완료: labelImage 형식 파일 정리 완료\n")


# Step 3: test → train 병합
print("[Step 3] Data/test → Data/train 병합")
print("-"*70)

test_dir = DATA_DIR / 'test'
train_dir = DATA_DIR / 'train'

if not test_dir.exists():
    print("  test 폴더가 존재하지 않습니다.")
else:
    test_images = list((test_dir / 'images').glob('*'))
    test_labels = list((test_dir / 'labels').glob('*.txt'))

    print(f"  test 폴더: 이미지 {len(test_images)}개, 라벨 {len(test_labels)}개")

    if len(test_images) == 0:
        print("  test 폴더가 비어있습니다.")
    else:
        # 파일명 중복 확인 및 이동
        moved_count = 0
        renamed_count = 0

        for img_file in test_images:
            lbl_file = test_dir / 'labels' / (img_file.stem + '.txt')

            # 목적지 경로
            dst_img = train_dir / 'images' / img_file.name
            dst_lbl = train_dir / 'labels' / lbl_file.name

            # 파일명 중복 시 번호 추가
            if dst_img.exists():
                counter = 1
                while True:
                    new_name = f"{img_file.stem}_test{counter}{img_file.suffix}"
                    dst_img = train_dir / 'images' / new_name
                    dst_lbl = train_dir / 'labels' / f"{img_file.stem}_test{counter}.txt"

                    if not dst_img.exists():
                        break
                    counter += 1

                renamed_count += 1

            # 이미지 이동
            shutil.move(str(img_file), str(dst_img))

            # 라벨 이동
            if lbl_file.exists():
                shutil.move(str(lbl_file), str(dst_lbl))

            moved_count += 1

        print(f"  {moved_count}개 파일 이동 완료 (중복으로 이름 변경: {renamed_count}개)")

        # test 폴더 삭제
        shutil.rmtree(test_dir)
        print(f"  test 폴더 삭제 완료")

print("\nStep 3 완료: test -> train 병합 완료\n")


# 최종 상태 확인
print("="*70)
print("최종 상태")
print("="*70)

print("\n[Data_polygon/]")
for split in ['train', 'valid']:
    split_dir = POLYGON_DIR / split
    if split_dir.exists():
        img_count = len(list((split_dir / 'images').glob('*')))
        lbl_count = len(list((split_dir / 'labels').glob('*.txt')))
        print(f"  {split}/: 이미지 {img_count}개, 라벨 {lbl_count}개")

print("\n[Data/]")
for split in ['train', 'valid']:
    split_dir = DATA_DIR / split
    if split_dir.exists():
        img_count = len(list((split_dir / 'images').glob('*')))
        lbl_count = len(list((split_dir / 'labels').glob('*.txt')))
        print(f"  {split}/: 이미지 {img_count}개, 라벨 {lbl_count}개")

print("\n" + "="*70)
print("정리 완료!")
print("="*70)
