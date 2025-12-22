"""
파일 번호 재정렬
fully_XXX.jpg, empty_XXX.jpg, combined_XXX.jpg를 0번부터 순서대로 재정렬
"""
import os
from pathlib import Path
import shutil

images_dir = Path('images')
labels_dir = Path('labels')

print("=" * 70)
print("파일 번호 재정렬 작업")
print("=" * 70)

# 클래스별로 파일 수집
classes = {
    'fully': sorted(images_dir.glob('fully_*.jpg')),
    'empty': sorted(images_dir.glob('empty_*.jpg')),
    'combined': sorted(images_dir.glob('combined_*.jpg'))
}

total_renamed = 0

for class_name, files in classes.items():
    print(f"\n{class_name} 클래스 처리 중... ({len(files)}개)")

    # 임시 이름으로 변경 (충돌 방지)
    temp_mappings = []
    for idx, old_path in enumerate(files):
        temp_name = f"temp_{class_name}_{idx}.jpg"
        temp_path = images_dir / temp_name

        # 라벨 파일도 확인
        old_label = labels_dir / f"{old_path.stem}.txt"
        temp_label = labels_dir / f"temp_{class_name}_{idx}.txt"

        # 임시 이름으로 변경
        shutil.move(str(old_path), str(temp_path))
        if old_label.exists():
            shutil.move(str(old_label), str(temp_label))
            temp_mappings.append((temp_path, temp_label, idx, True))
        else:
            temp_mappings.append((temp_path, None, idx, False))

    # 최종 이름으로 변경
    for temp_img, temp_label, idx, has_label in temp_mappings:
        final_name = f"{class_name}_{idx}.jpg"
        final_path = images_dir / final_name
        final_label_path = labels_dir / f"{class_name}_{idx}.txt"

        # 최종 이름으로 변경
        shutil.move(str(temp_img), str(final_path))
        if has_label and temp_label:
            shutil.move(str(temp_label), str(final_label_path))

        total_renamed += 1

        if idx < 5 or idx >= len(temp_mappings) - 2:
            label_status = "O" if has_label else "X"
            print(f"  [{idx:3d}] {final_name} (라벨: {label_status})")
        elif idx == 5:
            print(f"  ...")

print("\n" + "=" * 70)
print("작업 완료")
print("=" * 70)
print(f"총 {total_renamed}개 파일 번호 재정렬 완료")

# 최종 확인
print("\n최종 파일 개수:")
for class_name in ['fully', 'empty', 'combined']:
    img_count = len(list(images_dir.glob(f'{class_name}_*.jpg')))
    label_count = len(list(labels_dir.glob(f'{class_name}_*.txt')))
    print(f"  {class_name}: 이미지 {img_count}개, 라벨 {label_count}개")

# 전체 통계
total_images = len(list(images_dir.glob('*.jpg')))
total_labels = len(list(labels_dir.glob('*.txt')))
print(f"\n전체: 이미지 {total_images}개, 라벨 {total_labels}개")
print(f"라벨 없는 이미지: {total_images - total_labels}개")
