"""
라벨 품질 검증 및 분석 스크립트
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# 경로 설정
base_dir = Path(__file__).parent
images_dir = base_dir / 'images'
labels_dir = base_dir / 'labels'

class_names = ['fully', 'empty', 'combined']
colors = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 165, 255)
}

# 검증 결과 저장
issues = {
    'invalid_coords': [],
    'tiny_boxes': [],
    'huge_boxes': [],
    'empty_labels': [],
    'no_image': [],
    'malformed': []
}

# 통계 데이터
stats = {
    'box_areas': defaultdict(list),
    'box_aspect_ratios': defaultdict(list),
    'objects_per_image': [],
    'class_counts': defaultdict(int)
}

print("=" * 70)
print("라벨 품질 검증 시작")
print("=" * 70)

# 라벨 파일 검증
label_files = sorted(labels_dir.glob('*.txt'))
total_labels = len(label_files)

print(f"\n총 라벨 파일: {total_labels}개")
print("\n검증 중...")

for idx, label_file in enumerate(label_files, 1):
    if idx % 100 == 0:
        print(f"  진행: {idx}/{total_labels}")

    stem = label_file.stem
    img_file = images_dir / f"{stem}.jpg"

    if not img_file.exists():
        issues['no_image'].append(stem)
        continue

    img = cv2.imread(str(img_file))
    if img is None:
        continue
    img_h, img_w = img.shape[:2]

    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
    except:
        issues['malformed'].append(stem)
        continue

    if len(lines) == 0:
        issues['empty_labels'].append(stem)
        continue

    objects_count = 0

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

        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                0 < width <= 1 and 0 < height <= 1):
            issues['invalid_coords'].append(f"{stem} - coords out of range")
            continue

        area = width * height
        if area < 0.001:
            issues['tiny_boxes'].append(f"{stem} - area={area:.4f}")
        elif area > 0.95:
            issues['huge_boxes'].append(f"{stem} - area={area:.4f}")

        stats['box_areas'][class_id].append(area)
        stats['box_aspect_ratios'][class_id].append(width / height if height > 0 else 0)
        stats['class_counts'][class_id] += 1
        objects_count += 1

    stats['objects_per_image'].append(objects_count)

print("\n" + "=" * 70)
print("검증 완료")
print("=" * 70)

print("\n[발견된 이슈]")
total_issues = 0
for issue_type, issue_list in issues.items():
    count = len(issue_list)
    total_issues += count
    if count > 0:
        print(f"  ❌ {issue_type}: {count}개")
        if count <= 5:
            for item in issue_list:
                print(f"     - {item}")
        else:
            for item in issue_list[:5]:
                print(f"     - {item}")
            print(f"     ... 외 {count-5}개")

if total_issues == 0:
    print("  [OK] 이슈 없음!")

print("\n" + "=" * 70)
print("[데이터 통계]")
print("=" * 70)

print("\n1. 클래스별 객체 수:")
for cls_id in sorted(stats['class_counts'].keys()):
    count = stats['class_counts'][cls_id]
    print(f"   {class_names[cls_id]}: {count}개")

print(f"\n2. 이미지당 평균 객체 수: {np.mean(stats['objects_per_image']):.2f}개")
print(f"   최소: {np.min(stats['objects_per_image'])}개")
print(f"   최대: {np.max(stats['objects_per_image'])}개")

print("\n3. 클래스별 박스 면적 통계:")
for cls_id in sorted(stats['box_areas'].keys()):
    areas = stats['box_areas'][cls_id]
    print(f"   {class_names[cls_id]}:")
    print(f"     평균: {np.mean(areas):.4f}")
    print(f"     중앙값: {np.median(areas):.4f}")
    print(f"     표준편차: {np.std(areas):.4f}")
    print(f"     범위: {np.min(areas):.4f} ~ {np.max(areas):.4f}")

print("\n4. 클래스별 가로세로 비율:")
for cls_id in sorted(stats['box_aspect_ratios'].keys()):
    ratios = stats['box_aspect_ratios'][cls_id]
    print(f"   {class_names[cls_id]}:")
    print(f"     평균: {np.mean(ratios):.2f}")
    print(f"     중앙값: {np.median(ratios):.2f}")

print("\n5. 이상치 탐지 (박스 면적 기준 - IQR method):")
for cls_id in sorted(stats['box_areas'].keys()):
    areas = np.array(stats['box_areas'][cls_id])
    q1, q3 = np.percentile(areas, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = areas[(areas < lower_bound) | (areas > upper_bound)]
    print(f"   {class_names[cls_id]}: {len(outliers)}개 이상치")
    if len(outliers) > 0:
        print(f"     범위: {outliers.min():.4f} ~ {outliers.max():.4f}")

# 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for cls_id in sorted(stats['box_areas'].keys()):
    areas = stats['box_areas'][cls_id]
    plt.hist(areas, bins=50, alpha=0.6, label=class_names[cls_id])
plt.xlabel('Box Area (normalized)')
plt.ylabel('Frequency')
plt.title('Box Area Distribution by Class')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
for cls_id in sorted(stats['box_aspect_ratios'].keys()):
    ratios = stats['box_aspect_ratios'][cls_id]
    plt.hist(ratios, bins=50, alpha=0.6, label=class_names[cls_id])
plt.xlabel('Aspect Ratio (width/height)')
plt.ylabel('Frequency')
plt.title('Aspect Ratio Distribution by Class')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(stats['objects_per_image'], bins=range(0, max(stats['objects_per_image'])+2),
         edgecolor='black', alpha=0.7)
plt.xlabel('Objects per Image')
plt.ylabel('Frequency')
plt.title('Objects per Image Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(base_dir / 'label_statistics.png', dpi=150, bbox_inches='tight')
print(f"\n[통계 그래프 저장: label_statistics.png]")

# 샘플 이미지 시각화
print("\n" + "=" * 70)
print("[샘플 이미지 시각화 (각 클래스별 5장씩)]")
print("=" * 70)

sample_output_dir = base_dir / 'label_quality_samples'
sample_output_dir.mkdir(exist_ok=True)

class_files = {0: [], 1: [], 2: []}
for label_file in label_files:
    stem = label_file.stem
    with open(label_file, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        continue

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            class_files[class_id].append(stem)
            break

samples_per_class = 5
for cls_id in [0, 1, 2]:
    print(f"\n{class_names[cls_id]} 샘플:")

    available = list(set(class_files[cls_id]))
    if len(available) < samples_per_class:
        samples = available
    else:
        samples = random.sample(available, samples_per_class)

    for i, stem in enumerate(samples, 1):
        img_file = images_dir / f"{stem}.jpg"
        label_file = labels_dir / f"{stem}.txt"

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)

                color = colors[class_id]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label = class_names[class_id]
                cv2.putText(img, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = sample_output_dir / f"{class_names[cls_id]}_{i}_{stem}.jpg"
        cv2.imwrite(str(output_path), img)
        print(f"  [OK] {stem}.jpg")

print(f"\n[샘플 이미지 저장: label_quality_samples/]")

print("\n" + "=" * 70)
print("라벨 품질 검증 완료!")
print("=" * 70)
