"""
yolo11n_240 (기존) vs 새로 추가된 데이터 비교
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

base_dir = Path(__file__).parent
labels_dir = base_dir / 'labels'

class_names = ['fully', 'empty', 'combined']

# 240개 이하는 기존, 241 이상은 신규로 가정
OLD_THRESHOLD = 240

old_stats = {
    'box_areas': defaultdict(list),
    'class_counts': defaultdict(int),
    'files': defaultdict(list)
}

new_stats = {
    'box_areas': defaultdict(list),
    'class_counts': defaultdict(int),
    'files': defaultdict(list)
}

print("=" * 70)
print("기존 데이터 vs 새 데이터 비교")
print("=" * 70)

label_files = sorted(labels_dir.glob('*.txt'))

for label_file in label_files:
    stem = label_file.stem

    # 파일 이름에서 숫자 추출
    try:
        # fully_123, empty_45, combined_78 형식
        parts = stem.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            file_num = int(parts[1])
            is_old = file_num <= OLD_THRESHOLD
        else:
            # 숫자 추출 실패 시 신규로 간주
            is_old = False
    except:
        is_old = False

    # 라벨 읽기
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        continue

    if len(lines) == 0:
        continue

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
        except:
            continue

        area = width * height

        if is_old:
            old_stats['box_areas'][class_id].append(area)
            old_stats['class_counts'][class_id] += 1
            old_stats['files'][class_id].append(stem)
        else:
            new_stats['box_areas'][class_id].append(area)
            new_stats['class_counts'][class_id] += 1
            new_stats['files'][class_id].append(stem)

print("\n[기존 데이터 (1~240)]")
print("-" * 70)
old_total = sum(old_stats['class_counts'].values())
for cls_id in sorted(old_stats['class_counts'].keys()):
    count = old_stats['class_counts'][cls_id]
    files = len(set(old_stats['files'][cls_id]))
    pct = (count / old_total * 100) if old_total > 0 else 0
    print(f"{class_names[cls_id]}: {count}개 객체 ({files}개 파일, {pct:.1f}%)")
print(f"총 객체: {old_total}개")

if len(old_stats['class_counts']) > 0:
    max_c = max(old_stats['class_counts'].values())
    min_c = min(old_stats['class_counts'].values())
    print(f"불균형 비율: {max_c/min_c:.2f}:1")

print("\n[새 데이터 (241~)]")
print("-" * 70)
new_total = sum(new_stats['class_counts'].values())
for cls_id in sorted(new_stats['class_counts'].keys()):
    count = new_stats['class_counts'][cls_id]
    files = len(set(new_stats['files'][cls_id]))
    pct = (count / new_total * 100) if new_total > 0 else 0
    print(f"{class_names[cls_id]}: {count}개 객체 ({files}개 파일, {pct:.1f}%)")
print(f"총 객체: {new_total}개")

if len(new_stats['class_counts']) > 0:
    max_c = max(new_stats['class_counts'].values())
    min_c = min(new_stats['class_counts'].values())
    print(f"불균형 비율: {max_c/min_c:.2f}:1")

# 박스 면적 비교
print("\n" + "=" * 70)
print("[박스 면적 품질 비교]")
print("=" * 70)

for cls_id in sorted(set(list(old_stats['box_areas'].keys()) + list(new_stats['box_areas'].keys()))):
    print(f"\n{class_names[cls_id]}:")

    if cls_id in old_stats['box_areas']:
        old_areas = np.array(old_stats['box_areas'][cls_id])
        print(f"  기존: 평균 {np.mean(old_areas):.4f}, 표준편차 {np.std(old_areas):.4f}")
        print(f"        범위 {np.min(old_areas):.4f} ~ {np.max(old_areas):.4f}")
    else:
        print(f"  기존: 데이터 없음")

    if cls_id in new_stats['box_areas']:
        new_areas = np.array(new_stats['box_areas'][cls_id])
        print(f"  신규: 평균 {np.mean(new_areas):.4f}, 표준편차 {np.std(new_areas):.4f}")
        print(f"        범위 {np.min(new_areas):.4f} ~ {np.max(new_areas):.4f}")

        # 품질 저하 경고
        if cls_id in old_stats['box_areas']:
            old_std = np.std(old_areas)
            new_std = np.std(new_areas)
            if new_std > old_std * 1.3:
                print(f"  >>> [경고] 신규 데이터의 표준편차가 {new_std/old_std:.2f}배 증가 (일관성 저하)")
    else:
        print(f"  신규: 데이터 없음")

# 불균형 증가 분석
print("\n" + "=" * 70)
print("[최종 분석 결과]")
print("=" * 70)

print("\n1. 데이터 증가량:")
print(f"   기존: {old_total}개 객체")
print(f"   신규 추가: {new_total}개 객체")
print(f"   총: {old_total + new_total}개 객체 (+{new_total/old_total*100:.1f}% 증가)")

print("\n2. 클래스 불균형 변화:")
if len(old_stats['class_counts']) > 0 and len(new_stats['class_counts']) > 0:
    old_max = max(old_stats['class_counts'].values())
    old_min = min(old_stats['class_counts'].values())
    old_ratio = old_max / old_min

    new_max = max(new_stats['class_counts'].values())
    new_min = min(new_stats['class_counts'].values())
    new_ratio = new_max / new_min

    # 합산
    total_counts = defaultdict(int)
    for cls_id in set(list(old_stats['class_counts'].keys()) + list(new_stats['class_counts'].keys())):
        total_counts[cls_id] = old_stats['class_counts'].get(cls_id, 0) + new_stats['class_counts'].get(cls_id, 0)

    total_max = max(total_counts.values())
    total_min = min(total_counts.values())
    total_ratio = total_max / total_min

    print(f"   기존 데이터: {old_ratio:.2f}:1")
    print(f"   신규 데이터: {new_ratio:.2f}:1")
    print(f"   합산 후: {total_ratio:.2f}:1")

    if total_ratio > old_ratio:
        print(f"   >>> [경고] 불균형이 악화됨 ({old_ratio:.2f} -> {total_ratio:.2f})")

# 신규 데이터 중 fully 비율
if new_total > 0:
    fully_pct_new = (new_stats['class_counts'].get(0, 0) / new_total * 100)
    print(f"\n3. 신규 데이터 중 fully 비율: {fully_pct_new:.1f}%")
    if fully_pct_new > 55:
        print(f"   >>> [경고] 신규 데이터가 fully에 과도하게 편향됨")

print("\n" + "=" * 70)
print("비교 분석 완료!")
print("=" * 70)
