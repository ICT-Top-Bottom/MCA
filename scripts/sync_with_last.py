"""
LAST 폴더에 남아있는 이미지만 유지
LAST에서 삭제된 이미지는 images, labels에서도 삭제
"""
import os
from pathlib import Path

images_dir = Path('images')
labels_dir = Path('labels')
last_dir = Path('LAST')

print("=" * 70)
print("LAST 폴더 기준으로 데이터셋 동기화")
print("=" * 70)

# LAST 폴더에 있는 이미지 파일명 수집
last_files = {f.name for f in last_dir.glob('*.jpg')}
print(f"\nLAST 폴더 이미지: {len(last_files)}개")

# images 폴더에 있는 이미지 파일명 수집
image_files = {f.name for f in images_dir.glob('*.jpg')}
print(f"images 폴더 이미지: {len(image_files)}개")

# 삭제해야 할 파일 (images에는 있지만 LAST에는 없는 것)
to_delete = image_files - last_files

print(f"\n삭제 대상: {len(to_delete)}개")

if len(to_delete) == 0:
    print("\n삭제할 파일이 없습니다!")
else:
    print("\n삭제 목록:")
    for filename in sorted(to_delete):
        print(f"  - {filename}")

    print("\n삭제 진행 중...")

    deleted_images = 0
    deleted_labels = 0

    for filename in to_delete:
        # 이미지 삭제
        img_path = images_dir / filename
        if img_path.exists():
            os.remove(img_path)
            deleted_images += 1
            print(f"  [이미지 삭제] {filename}")

        # 라벨 삭제
        label_filename = filename.replace('.jpg', '.txt')
        label_path = labels_dir / label_filename
        if label_path.exists():
            os.remove(label_path)
            deleted_labels += 1
            print(f"  [라벨 삭제] {label_filename}")

    print("\n" + "=" * 70)
    print("삭제 완료")
    print("=" * 70)
    print(f"삭제된 이미지: {deleted_images}개")
    print(f"삭제된 라벨: {deleted_labels}개")

# 최종 확인
print("\n최종 파일 개수:")
final_images = len(list(images_dir.glob('*.jpg')))
final_labels = len(list(labels_dir.glob('*.txt')))
last_count = len(list(last_dir.glob('*.jpg')))

print(f"  images: {final_images}개")
print(f"  labels: {final_labels}개")
print(f"  LAST: {last_count}개")

if final_images == final_labels == last_count:
    print("\n✅ 모든 폴더가 동기화되었습니다!")
else:
    print(f"\n⚠️  불일치 발생")
    print(f"    images-labels 차이: {abs(final_images - final_labels)}개")
    print(f"    images-LAST 차이: {abs(final_images - last_count)}개")

# 클래스별 개수
print("\n클래스별 최종 개수:")
for class_name in ['fully', 'empty', 'combined']:
    img_count = len(list(images_dir.glob(f'{class_name}_*.jpg')))
    label_count = len(list(labels_dir.glob(f'{class_name}_*.txt')))
    print(f"  {class_name}: 이미지 {img_count}개, 라벨 {label_count}개")
