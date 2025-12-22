"""
라벨 데이터 분석 (이미지 읽기 없이 라벨 파일만 분석)
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경
import matplotlib.pyplot as plt

base_dir = Path(__file__).parent
labels_dir = base_dir / 'labels'
images_dir = base_dir / 'images'

class_names = ['fully', 'empty', 'combined']

# 통계
stats = {
    'box_areas': defaultdict(list),
    'box_widths': defaultdict(list),
    'box_heights': defaultdict(list),
    'box_aspect_ratios': defaultdict(list),
    'objects_per_image': [],
    'class_counts': defaultdict(int),
    'class_per_file': defaultdict(list)
}

issues = {
    'invalid_coords': [],
    'tiny_boxes': [],
    'huge_boxes': [],
    'empty_labels': [],
    'malformed': []
}

print("=" * 70)
print("라벨 데이터 분석")
print("=" * 70)

label_files = sorted(labels_dir.glob('*.txt'))
print(f"\n총 라벨 파일: {len(label_files)}개\n")

# 라벨 분석
for idx, label_file in enumerate(label_files, 1):
    if idx % 100 == 0:
        print(f"진행: {idx}/{len(label_files)}")

    stem = label_file.stem

    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        issues['malformed'].append(stem)
        continue

    if len(lines) == 0:
        issues['empty_labels'].append(stem)
        continue

    objects_count = 0
    file_classes = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            issues['malformed'].append(f"{stem} - invalid format")
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
        except:
            issues['malformed'].append(f"{stem} - parse error")
            continue

        # 좌표 검증
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                0 < width <= 1 and 0 < height <= 1):
            issues['invalid_coords'].append(f"{stem}")
            continue

        # 박스 크기 검증
        area = width * height
        if area < 0.001:
            issues['tiny_boxes'].append(f"{stem} - area={area:.4f}")
        elif area > 0.95:
            issues['huge_boxes'].append(f"{stem} - area={area:.4f}")

        # 통계 수집
        stats['box_areas'][class_id].append(area)
        stats['box_widths'][class_id].append(width)
        stats['box_heights'][class_id].append(height)
        stats['box_aspect_ratios'][class_id].append(width / height if height > 0 else 0)
        stats['class_counts'][class_id] += 1
        file_classes.append(class_id)
        objects_count += 1

    stats['objects_per_image'].append(objects_count)
    for cls_id in file_classes:
        stats['class_per_file'][cls_id].append(stem)

print("\n" + "=" * 70)
print("분석 완료")
print("=" * 70)

# 이슈 출력
print("\n[발견된 이슈]")
total_issues = 0
for issue_type, issue_list in issues.items():
    count = len(set(issue_list))  # 중복 제거
    total_issues += count
    if count > 0:
        print(f"  X {issue_type}: {count}개")
        if count <= 3:
            for item in list(set(issue_list)):
                print(f"     - {item}")

if total_issues == 0:
    print("  [OK] 심각한 이슈 없음!")

# 통계 출력
print("\n" + "=" * 70)
print("[데이터 통계]")
print("=" * 70)

print("\n1. 클래스별 객체 수 (불균형 분석):")
total_objects = sum(stats['class_counts'].values())
for cls_id in sorted(stats['class_counts'].keys()):
    count = stats['class_counts'][cls_id]
    percentage = (count / total_objects * 100) if total_objects > 0 else 0
    print(f"   {class_names[cls_id]}: {count}개 ({percentage:.1f}%)")

print(f"   총 객체: {total_objects}개")

# 불균형 비율 계산
if len(stats['class_counts']) > 0:
    max_count = max(stats['class_counts'].values())
    min_count = min(stats['class_counts'].values())
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    print(f"\n   >> 클래스 불균형 비율: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 2.0:
        print(f"   >> [경고] 심각한 불균형 ({imbalance_ratio:.1f}배 차이)")

print(f"\n2. 이미지당 객체 수:")
if len(stats['objects_per_image']) > 0:
    print(f"   평균: {np.mean(stats['objects_per_image']):.2f}개")
    print(f"   중앙값: {np.median(stats['objects_per_image']):.0f}개")
    print(f"   범위: {np.min(stats['objects_per_image'])} ~ {np.max(stats['objects_per_image'])}개")

print("\n3. 클래스별 박스 면적 통계 (normalized):")
for cls_id in sorted(stats['box_areas'].keys()):
    areas = np.array(stats['box_areas'][cls_id])
    print(f"\n   {class_names[cls_id]}:")
    print(f"     평균: {np.mean(areas):.4f}")
    print(f"     중앙값: {np.median(areas):.4f}")
    print(f"     표준편차: {np.std(areas):.4f}")
    print(f"     범위: {np.min(areas):.4f} ~ {np.max(areas):.4f}")

print("\n4. 클래스별 가로세로 비율 (width/height):")
for cls_id in sorted(stats['box_aspect_ratios'].keys()):
    ratios = np.array(stats['box_aspect_ratios'][cls_id])
    print(f"   {class_names[cls_id]}: 평균 {np.mean(ratios):.2f}, 중앙값 {np.median(ratios):.2f}")

print("\n5. 이상치 탐지 (IQR method, 박스 면적 기준):")
outliers_summary = {}
for cls_id in sorted(stats['box_areas'].keys()):
    areas = np.array(stats['box_areas'][cls_id])
    q1, q3 = np.percentile(areas, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = areas[(areas < lower_bound) | (areas > upper_bound)]
    outliers_summary[cls_id] = len(outliers)
    print(f"   {class_names[cls_id]}: {len(outliers)}개 이상치 ({len(outliers)/len(areas)*100:.1f}%)")

# 시각화
fig = plt.figure(figsize=(16, 10))

# 1. 클래스별 박스 면적 분포
plt.subplot(2, 3, 1)
for cls_id in sorted(stats['box_areas'].keys()):
    areas = stats['box_areas'][cls_id]
    plt.hist(areas, bins=50, alpha=0.6, label=class_names[cls_id])
plt.xlabel('Box Area (normalized)')
plt.ylabel('Frequency')
plt.title('Box Area Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 클래스별 가로세로 비율
plt.subplot(2, 3, 2)
for cls_id in sorted(stats['box_aspect_ratios'].keys()):
    ratios = stats['box_aspect_ratios'][cls_id]
    plt.hist(ratios, bins=50, alpha=0.6, label=class_names[cls_id])
plt.xlabel('Aspect Ratio (w/h)')
plt.ylabel('Frequency')
plt.title('Aspect Ratio Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. 이미지당 객체 수
plt.subplot(2, 3, 3)
plt.hist(stats['objects_per_image'], bins=range(0, max(stats['objects_per_image'])+2),
         edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Objects per Image')
plt.ylabel('Frequency')
plt.title('Objects per Image Distribution')
plt.grid(True, alpha=0.3)

# 4. 클래스 불균형 시각화
plt.subplot(2, 3, 4)
classes = [class_names[i] for i in sorted(stats['class_counts'].keys())]
counts = [stats['class_counts'][i] for i in sorted(stats['class_counts'].keys())]
colors_bar = ['green', 'blue', 'orange']
plt.bar(classes, counts, color=colors_bar, alpha=0.7, edgecolor='black')
plt.ylabel('Count')
plt.title('Class Imbalance')
plt.grid(True, alpha=0.3, axis='y')
for i, (c, count) in enumerate(zip(classes, counts)):
    plt.text(i, count, f'{count}\n({count/sum(counts)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# 5. 박스 크기 boxplot
plt.subplot(2, 3, 5)
box_data = [stats['box_areas'][i] for i in sorted(stats['box_areas'].keys())]
plt.boxplot(box_data, labels=classes)
plt.ylabel('Box Area (normalized)')
plt.title('Box Area Distribution (Boxplot)')
plt.grid(True, alpha=0.3, axis='y')

# 6. 이상치 비율
plt.subplot(2, 3, 6)
outlier_pcts = [outliers_summary[i]/len(stats['box_areas'][i])*100
                for i in sorted(outliers_summary.keys())]
plt.bar(classes, outlier_pcts, color=colors_bar, alpha=0.7, edgecolor='black')
plt.ylabel('Outlier Percentage (%)')
plt.title('Outlier Ratio by Class')
plt.grid(True, alpha=0.3, axis='y')
for i, (c, pct) in enumerate(zip(classes, outlier_pcts)):
    plt.text(i, pct, f'{pct:.1f}%', ha='center', va='bottom')

plt.tight_layout()
output_path = base_dir / 'label_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[분석 결과 저장: {output_path.name}]")

# 클래스별 파일 리스트 (샘플)
print("\n" + "=" * 70)
print("[클래스별 파일 샘플 (각 5개씩)]")
print("=" * 70)
for cls_id in sorted(stats['class_per_file'].keys()):
    files = list(set(stats['class_per_file'][cls_id]))
    print(f"\n{class_names[cls_id]} ({len(files)}개 파일):")
    for f in files[:5]:
        print(f"  - {f}.jpg")

print("\n" + "=" * 70)
print("라벨 분석 완료!")
print("=" * 70)
