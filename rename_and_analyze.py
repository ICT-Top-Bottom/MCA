"""
이미지 파일명 간결하게 정리 및 클래스별 통계
"""
from pathlib import Path
from collections import Counter
import shutil

BASE_DIR = Path(r'c:\Users\24457\OneDrive\바탕 화면\MCA')

print("="*70)
print("파일명 정리 및 클래스별 통계")
print("="*70)

# Data와 Data_polygon 모두 처리
for dataset_name in ['Data', 'Data_polygon']:
    dataset_dir = BASE_DIR / dataset_name

    if not dataset_dir.exists():
        continue

    print(f"\n{'='*70}")
    print(f"{dataset_name}/")
    print("="*70)

    for split in ['train', 'valid', 'test']:
        split_dir = dataset_dir / split

        if not split_dir.exists():
            continue

        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'

        if not images_dir.exists():
            continue

        print(f"\n[{split}]")
        print("-"*70)

        # 이미지 파일 목록
        image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))

        # 클래스별 카운터
        class_counter = Counter()
        total_objects = 0

        # 파일명 변경 카운터
        rename_counter = 1

        for img_file in image_files:
            # 라벨 파일 찾기
            label_file = labels_dir / (img_file.stem + '.txt')

            if not label_file.exists():
                # .jpg를 _jpg로 찾기
                label_file = labels_dir / (img_file.stem.replace('.jpg', '_jpg') + '.txt')

            if not label_file.exists():
                # 다른 패턴 시도
                possible_labels = list(labels_dir.glob(f"{img_file.stem}*.txt"))
                if possible_labels:
                    label_file = possible_labels[0]

            # 라벨에서 클래스 카운팅
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counter[class_id] += 1
                            total_objects += 1

            # 새 파일명 생성
            extension = img_file.suffix
            new_img_name = f"{split}_{rename_counter:04d}{extension}"
            new_lbl_name = f"{split}_{rename_counter:04d}.txt"

            new_img_path = images_dir / new_img_name
            new_lbl_path = labels_dir / new_lbl_name

            # 파일명이 이미 같으면 건너뛰기
            if img_file.name != new_img_name:
                # 임시 파일명으로 이동 (충돌 방지)
                temp_img = images_dir / f"temp_{rename_counter}{extension}"
                temp_lbl = labels_dir / f"temp_{rename_counter}.txt"

                shutil.move(str(img_file), str(temp_img))
                if label_file.exists():
                    shutil.move(str(label_file), str(temp_lbl))

                # 최종 이름으로 변경
                shutil.move(str(temp_img), str(new_img_path))
                if temp_lbl.exists():
                    shutil.move(str(temp_lbl), str(new_lbl_path))

            rename_counter += 1

        # 통계 출력
        print(f"이미지 개수: {len(image_files)}개")
        print(f"전체 객체 수: {total_objects}개")
        print(f"\n클래스별 객체 수:")

        class_names = {0: 'combined', 1: 'empty', 2: 'fully'}
        for class_id in sorted(class_counter.keys()):
            count = class_counter[class_id]
            class_name = class_names.get(class_id, f'class_{class_id}')
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"  {class_name} (class {class_id}): {count}개 ({percentage:.1f}%)")

print("\n" + "="*70)
print("파일명 정리 및 분석 완료!")
print("="*70)
