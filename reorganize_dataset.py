"""
데이터셋 재구성 스크립트:
1. 현재 비율 확인
2. Test를 Train에 합쳐서 8:2 비율로 재조정
3. 파일명 통일 (0001.jpg 형식)
4. roboflow와 roboflow_visualized_polygon 둘 다 처리
"""
import shutil
from pathlib import Path
from collections import Counter

def count_files(folder):
    """폴더 내 파일 개수"""
    images = list((folder / 'images').glob('*.jpg'))
    return len(images)

def check_current_ratio():
    """현재 비율 확인"""
    roboflow = Path('roboflow')

    train_count = count_files(roboflow / 'train')
    valid_count = count_files(roboflow / 'valid')
    test_count = count_files(roboflow / 'test')
    total = train_count + valid_count + test_count

    print("="*60)
    print("Current Dataset Ratio")
    print("="*60)
    print(f"Train: {train_count} ({train_count/total*100:.1f}%)")
    print(f"Valid: {valid_count} ({valid_count/total*100:.1f}%)")
    print(f"Test:  {test_count} ({test_count/total*100:.1f}%)")
    print(f"Total: {total}")
    print("="*60)

    return train_count, valid_count, test_count, total

def merge_test_to_train(base_folder):
    """Test 폴더를 Train에 합치기"""
    train_images = base_folder / 'train' / 'images'
    train_labels = base_folder / 'train' / 'labels'
    test_images = base_folder / 'test' / 'images'
    test_labels = base_folder / 'test' / 'labels'

    if not test_images.exists():
        print(f"  No test folder in {base_folder.name}")
        return 0

    moved = 0
    for img_file in test_images.glob('*.jpg'):
        label_file = test_labels / f"{img_file.stem}.txt"

        # 이미지 이동
        shutil.move(str(img_file), str(train_images / img_file.name))

        # 라벨 이동
        if label_file.exists():
            shutil.move(str(label_file), str(train_labels / label_file.name))

        moved += 1

    # test 폴더 삭제
    try:
        shutil.rmtree(test_images)
        shutil.rmtree(test_labels)
        (base_folder / 'test').rmdir()
    except:
        pass

    return moved

def rename_files(folder, start_index):
    """파일명 통일 (0001.jpg 형식)"""
    images_dir = folder / 'images'
    labels_dir = folder / 'labels'

    if not images_dir.exists():
        return start_index

    image_files = sorted(list(images_dir.glob('*.jpg')))

    current_index = start_index
    renamed = 0

    for img_file in image_files:
        old_stem = img_file.stem
        new_name = f"{current_index:04d}"

        # 새 경로
        new_img = images_dir / f"{new_name}.jpg"
        new_label = labels_dir / f"{new_name}.txt"
        old_label = labels_dir / f"{old_stem}.txt"

        # 이미 같은 이름이면 스킵
        if img_file == new_img:
            current_index += 1
            continue

        # 임시 파일명으로 변경 (충돌 방지)
        temp_img = images_dir / f"temp_{current_index}.jpg"
        temp_label = labels_dir / f"temp_{current_index}.txt"

        img_file.rename(temp_img)
        if old_label.exists():
            old_label.rename(temp_label)

        # 최종 이름으로 변경
        temp_img.rename(new_img)
        if temp_label.exists():
            temp_label.rename(new_label)

        renamed += 1
        current_index += 1

    return current_index

# 1. 현재 상태 확인
train_count, valid_count, test_count, total = check_current_ratio()

# 2. roboflow 폴더 처리
print("\n[1/2] Processing roboflow folder...")
roboflow = Path('roboflow')

print("  - Merging test to train...")
moved = merge_test_to_train(roboflow)
print(f"    Moved {moved} files from test to train")

print("  - Renaming train files...")
train_end = rename_files(roboflow / 'train', 1)
print(f"    Train: 0001 ~ {train_end-1:04d}")

print("  - Renaming valid files...")
valid_end = rename_files(roboflow / 'valid', train_end)
print(f"    Valid: {train_end:04d} ~ {valid_end-1:04d}")

# 3. roboflow_visualized_polygon 폴더 처리
print("\n[2/2] Processing roboflow_visualized_polygon folder...")
polygon = Path('roboflow_visualized_polygon')

if polygon.exists():
    print("  - Merging test to train...")
    moved = merge_test_to_train(polygon)
    print(f"    Moved {moved} files from test to train")

    print("  - Renaming train files...")
    train_end = rename_files(polygon / 'train', 1)
    print(f"    Train: 0001 ~ {train_end-1:04d}")

    print("  - Renaming valid files...")
    valid_end = rename_files(polygon / 'valid', train_end)
    print(f"    Valid: {train_end:04d} ~ {valid_end-1:04d}")
else:
    print("  - Folder not found, skipping")

# 4. 최종 결과
print("\n" + "="*60)
print("Final Dataset Ratio")
print("="*60)

final_train = count_files(roboflow / 'train')
final_valid = count_files(roboflow / 'valid')
final_total = final_train + final_valid

print(f"Train: {final_train} ({final_train/final_total*100:.1f}%)")
print(f"Valid: {final_valid} ({final_valid/final_total*100:.1f}%)")
print(f"Total: {final_total}")
print("="*60)
print("\nDone!")
