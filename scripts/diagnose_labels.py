"""
라벨 파일 불일치 문제 진단
Box와 Segment 개수가 맞지 않는 파일 찾기
"""
from pathlib import Path
from collections import defaultdict

def count_annotations(label_file):
    """라벨 파일의 annotation 개수 세기"""
    boxes = 0
    segments = 0

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # 5개 필드 = bbox (class x_center y_center width height)
            if len(parts) == 5:
                boxes += 1
            # 5개 이상 = polygon segmentation
            else:
                segments += 1

    return boxes, segments

def diagnose_dataset(dataset_path):
    """데이터셋 전체 진단"""
    dataset_path = Path(dataset_path)

    print("=" * 70)
    print("Label Dataset Diagnosis")
    print("=" * 70)

    total_boxes = 0
    total_segments = 0
    problematic_files = []

    for split in ['train', 'valid']:
        labels_dir = dataset_path / split / 'labels'

        if not labels_dir.exists():
            print(f"\n{split}: Directory not found")
            continue

        print(f"\n{split.upper()}:")
        print("-" * 70)

        split_boxes = 0
        split_segments = 0
        split_problems = []

        for label_file in labels_dir.glob('*.txt'):
            boxes, segments = count_annotations(label_file)
            split_boxes += boxes
            split_segments += segments

            # bbox만 있거나 혼합 형식인 경우
            if boxes > 0 and segments == 0:
                split_problems.append({
                    'file': label_file.name,
                    'boxes': boxes,
                    'segments': segments,
                    'issue': 'bbox_only'
                })
            elif boxes > 0 and segments > 0:
                split_problems.append({
                    'file': label_file.name,
                    'boxes': boxes,
                    'segments': segments,
                    'issue': 'mixed'
                })

        print(f"  Total boxes:    {split_boxes}")
        print(f"  Total segments: {split_segments}")
        print(f"  Total files:    {len(list(labels_dir.glob('*.txt')))}")
        print(f"  Problem files:  {len(split_problems)}")

        total_boxes += split_boxes
        total_segments += split_segments
        problematic_files.extend(split_problems)

        if split_problems:
            print(f"\n  Problematic files in {split}:")
            for problem in split_problems[:10]:  # 처음 10개만 출력
                print(f"    - {problem['file']}: {problem['boxes']} boxes, {problem['segments']} segments ({problem['issue']})")
            if len(split_problems) > 10:
                print(f"    ... and {len(split_problems) - 10} more")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total boxes:           {total_boxes}")
    print(f"Total segments:        {total_segments}")
    print(f"Total problem files:   {len(problematic_files)}")
    print(f"Difference:            {abs(total_boxes - total_segments)}")

    if total_boxes != total_segments:
        print("\n⚠️  WARNING: Box and segment counts do NOT match!")
        print("This will cause Ultralytics to drop ALL segments and use only boxes.")

        issue_types = defaultdict(int)
        for p in problematic_files:
            issue_types[p['issue']] += 1

        print("\nIssue breakdown:")
        for issue, count in issue_types.items():
            print(f"  - {issue}: {count} files")
    else:
        print("\n✅ All annotations are properly formatted!")

    return problematic_files

if __name__ == '__main__':
    problems = diagnose_dataset('roboflow')

    # 문제 파일 리스트 저장
    if problems:
        with open('problematic_labels.txt', 'w') as f:
            for p in problems:
                f.write(f"{p['file']}\t{p['boxes']}\t{p['segments']}\t{p['issue']}\n")
        print(f"\n💾 Problem files saved to: problematic_labels.txt")
