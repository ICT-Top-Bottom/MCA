"""
데이터셋 무결성 체크 및 파일명 분석
"""
import os
from pathlib import Path
from collections import defaultdict

images_dir = Path('images')
labels_dir = Path('labels')

# 이미지 파일 수집
image_files = sorted([f.name for f in images_dir.glob('*.jpg')])
label_files = sorted([f.name for f in labels_dir.glob('*.txt')])

print("=" * 70)
print("데이터셋 무결성 체크")
print("=" * 70)

print(f"\n총 이미지 개수: {len(image_files)}")
print(f"총 라벨 개수: {len(label_files)}")

# 매칭 확인
image_stems = {Path(f).stem for f in image_files}
label_stems = {Path(f).stem for f in label_files}

matched = image_stems & label_stems
only_images = image_stems - label_stems
only_labels = label_stems - image_stems

print(f"\n매칭된 쌍: {len(matched)}")
print(f"이미지만 있음: {len(only_images)}")
print(f"라벨만 있음: {len(only_labels)}")

if only_images:
    print(f"\n라벨 없는 이미지 ({len(only_images)}개):")
    for stem in sorted(only_images)[:10]:
        print(f"  {stem}.jpg")
    if len(only_images) > 10:
        print(f"  ... and {len(only_images) - 10} more")

if only_labels:
    print(f"\n이미지 없는 라벨 ({len(only_labels)}개):")
    for stem in sorted(only_labels)[:10]:
        print(f"  {stem}.txt")
    if len(only_labels) > 10:
        print(f"  ... and {len(only_labels) - 10} more")

# 파일명 패턴 분석
print("\n" + "=" * 70)
print("파일명 패턴 분석")
print("=" * 70)

patterns = defaultdict(list)

for img in image_files:
    if img.startswith('fully_') and 'BOOK' not in img:
        patterns['fully_XXX.jpg'].append(img)
    elif img.startswith('fully_') and 'BOOK' in img:
        patterns['fully_XXX_BOOK.jpg'].append(img)
    elif img.startswith('empty_'):
        patterns['empty_XXX.jpg'].append(img)
    elif img.startswith('e_') and 'ver2' in img:
        patterns['e_XX_ver2.jpg'].append(img)
    elif img.startswith('combined_'):
        patterns['combined_XXX.jpg'].append(img)
    else:
        patterns['other'].append(img)

print(f"\n파일명 패턴별 개수:")
for pattern, files in sorted(patterns.items()):
    print(f"  {pattern}: {len(files)}개")
    if len(files) <= 5:
        for f in files:
            print(f"    - {f}")

# 통일 필요한 파일 확인
print("\n" + "=" * 70)
print("통일 필요한 파일명")
print("=" * 70)

need_rename = []

# BOOK 파일
for img in patterns['fully_XXX_BOOK.jpg']:
    need_rename.append(('BOOK 제거', img))

# e_ver2 파일
for img in patterns['e_XX_ver2.jpg']:
    need_rename.append(('e_ver2 → empty', img))

print(f"\n통일 필요: {len(need_rename)}개")
for reason, filename in need_rename[:20]:
    print(f"  [{reason}] {filename}")

if len(need_rename) > 20:
    print(f"  ... and {len(need_rename) - 20} more")

print("\n" + "=" * 70)
print("요약")
print("=" * 70)
print(f"✓ 총 이미지: {len(image_files)}")
print(f"✓ 총 라벨: {len(label_files)}")
print(f"✓ 매칭된 쌍: {len(matched)}")
print(f"✓ 통일 필요: {len(need_rename)}개")

if len(matched) == len(image_files) == len(label_files):
    print("\n✅ 모든 이미지-라벨 쌍이 일치합니다!")
else:
    print(f"\n⚠️  이미지-라벨 쌍이 일치하지 않습니다!")
