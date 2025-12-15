"""
Data_polygon/test에 누락된 이미지 40개 복원
"""
from pathlib import Path
import shutil

BASE_DIR = Path(r'c:\Users\24457\OneDrive\바탕 화면\MCA')
DATA_TRAIN = BASE_DIR / 'Data' / 'train' / 'images'
POLYGON_TEST_IMAGES = BASE_DIR / 'Data_polygon' / 'test' / 'images'
POLYGON_TEST_LABELS = BASE_DIR / 'Data_polygon' / 'test' / 'labels'

print("="*70)
print("Data_polygon/test 이미지 복원")
print("="*70)

# test 라벨 파일 목록
label_files = list(POLYGON_TEST_LABELS.glob('*.txt'))
print(f"\ntest 라벨 개수: {len(label_files)}개")

# Data/train에서 마지막 40개 이미지 찾기 (test에서 병합된 것들)
train_images = sorted(DATA_TRAIN.glob('*.jpg')) + sorted(DATA_TRAIN.glob('*.png'))
print(f"Data/train 이미지 개수: {len(train_images)}개")

# 마지막 40개가 test에서 온 것
test_images = train_images[-40:]
print(f"\n복사할 이미지: 마지막 {len(test_images)}개")

copied_count = 0
for idx, img_file in enumerate(test_images, 1):
    # 새 이름으로 복사
    new_name = f"test_{idx:04d}{img_file.suffix}"
    dst = POLYGON_TEST_IMAGES / new_name

    shutil.copy(str(img_file), str(dst))
    copied_count += 1

print(f"\n{copied_count}개 이미지 복사 완료!")

# 최종 확인
final_img_count = len(list(POLYGON_TEST_IMAGES.glob('*')))
final_lbl_count = len(list(POLYGON_TEST_LABELS.glob('*.txt')))

print("\n" + "="*70)
print("복원 완료!")
print("="*70)
print(f"Data_polygon/test/images: {final_img_count}개")
print(f"Data_polygon/test/labels: {final_lbl_count}개")
print("="*70)
