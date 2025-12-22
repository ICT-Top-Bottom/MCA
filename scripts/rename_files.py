"""
파일명 통일 작업
1. fully_XXX-BOOK-XXX.jpg → fully_XXX.jpg
2. e_XX_ver2.jpg → empty_XX.jpg
"""
import os
import shutil
from pathlib import Path

images_dir = Path('images')
labels_dir = Path('labels')

print("=" * 70)
print("파일명 통일 작업")
print("=" * 70)

renamed_count = 0
errors = []

# 1. BOOK 파일 이름 변경
print("\n1. BOOK 파일 처리...")
book_files = list(images_dir.glob('*-BOOK-*.jpg'))

for old_path in book_files:
    # fully_138-BOOK-7F1IFULU9P.jpg → fully_138.jpg
    parts = old_path.stem.split('-BOOK-')
    if len(parts) == 2:
        new_name = f"{parts[0]}.jpg"
        new_path = images_dir / new_name

        # 라벨 파일도 확인
        old_label = labels_dir / f"{old_path.stem}.txt"
        new_label = labels_dir / f"{parts[0]}.txt"

        print(f"  {old_path.name} → {new_name}")

        try:
            # 이미지 이름 변경
            shutil.move(str(old_path), str(new_path))

            # 라벨 파일도 있으면 이름 변경
            if old_label.exists():
                shutil.move(str(old_label), str(new_label))
                print(f"    라벨도 변경: {old_label.name} → {new_label.name}")

            renamed_count += 1
        except Exception as e:
            errors.append(f"  Error: {old_path.name} - {e}")

# 2. e_ver2 파일 이름 변경
print("\n2. e_ver2 파일 처리...")
e_files = list(images_dir.glob('e_*_ver2.jpg'))

for old_path in e_files:
    # e_40_ver2.jpg → empty_40.jpg
    num = old_path.stem.split('_')[1]  # '40'
    new_name = f"empty_{num}.jpg"
    new_path = images_dir / new_name

    # 라벨 파일도 확인
    old_label = labels_dir / f"{old_path.stem}.txt"
    new_label = labels_dir / f"empty_{num}.txt"

    print(f"  {old_path.name} → {new_name}")

    try:
        # 이미지 이름 변경
        shutil.move(str(old_path), str(new_path))

        # 라벨 파일도 있으면 이름 변경
        if old_label.exists():
            shutil.move(str(old_label), str(new_label))
            print(f"    라벨도 변경: {old_label.name} → {new_label.name}")

        renamed_count += 1
    except Exception as e:
        errors.append(f"  Error: {old_path.name} - {e}")

print("\n" + "=" * 70)
print("작업 완료")
print("=" * 70)
print(f"✓ 총 {renamed_count}개 파일 이름 변경")

if errors:
    print(f"\n⚠️  {len(errors)}개 오류 발생:")
    for error in errors:
        print(error)
else:
    print("\n✅ 모든 파일 이름 변경 성공!")

# 최종 확인
print("\n최종 파일 개수:")
image_count = len(list(images_dir.glob('*.jpg')))
label_count = len(list(labels_dir.glob('*.txt')))
print(f"  이미지: {image_count}개")
print(f"  라벨: {label_count}개")
