"""
YOLO11n Multi-Scale 모델로 testImage 폴더 이미지 추론
멀리 있는 작은 카트 탐지 개선 확인
"""
from ultralytics import YOLO
from pathlib import Path
import os

# 모델 로드
model_path = Path(__file__).parent / 'yolo11n_multiscale' / 'best.pt'
model = YOLO(str(model_path))

# 테스트 이미지 디렉토리
test_dir = Path(__file__).parent / 'testImage'
output_dir = Path(__file__).parent / 'test_multiscale_results'

print("=" * 70)
print("YOLO11n Multi-Scale 모델 테스트")
print("=" * 70)
print(f"\nModel: {model_path}")
print(f"Test images: {test_dir}")
print(f"Output: {output_dir}")
print("\n🎯 목적: 멀리 있는 작은 카트 탐지 개선 확인")
print("   - 이미지 크기: 1024px (기존 640px 대비 2.56배)")
print("   - Multi-scale training 적용")

# 테스트 이미지 개수 확인
test_images = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
print(f"\n총 {len(test_images)}개 이미지 테스트")

# 추론 실행 (1024px)
print("\n" + "=" * 70)
print("추론 시작 (1024px)...")
print("=" * 70)

results = model.predict(
    source=str(test_dir),
    save=True,
    project=str(output_dir.parent),
    name=output_dir.name,
    exist_ok=True,
    conf=0.25,
    iou=0.45,
    imgsz=1024,  # 큰 이미지 크기로 추론
    device='cpu',
    verbose=True
)

print("\n" + "=" * 70)
print("추론 완료!")
print("=" * 70)
print(f"\n결과 저장 위치: {output_dir}/")

# 클래스별 탐지 통계
class_names = {0: 'fully', 1: 'empty', 2: 'combined'}
class_counts = {0: 0, 1: 0, 2: 0}

for result in results:
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls = int(box.cls[0])
            class_counts[cls] += 1

print("\n클래스별 탐지 결과:")
for cls_id, count in sorted(class_counts.items()):
    print(f"  {class_names[cls_id]}: {count}개")

total_detections = sum(class_counts.values())
print(f"\n총 탐지된 객체: {total_detections}개")

# 이미지별 결과 출력
print("\n" + "=" * 70)
print("이미지별 탐지 결과")
print("=" * 70)

for i, result in enumerate(results):
    img_name = Path(test_images[i]).name
    num_detections = len(result.boxes) if result.boxes is not None else 0

    if num_detections > 0:
        detections_str = []
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections_str.append(f"{class_names[cls]}({conf:.2f})")
        print(f"\n{img_name}: {num_detections}개 탐지")
        print(f"  → {', '.join(detections_str)}")
    else:
        print(f"\n{img_name}: 탐지 없음")

# Baseline 비교 (balanced_v2 결과)
print("\n" + "=" * 70)
print("Baseline Comparison (yolo11n_balanced_v2)")
print("=" * 70)
print("\nBaseline 결과:")
print("  - Total: 10개 (fully: 3, empty: 3, combined: 4)")
print(f"\nMulti-Scale 결과:")
print(f"  - Total: {total_detections}개 (fully: {class_counts[0]}, empty: {class_counts[1]}, combined: {class_counts[2]})")

if total_detections > 10:
    improvement = ((total_detections - 10) / 10) * 100
    print(f"\n✅ 개선: +{total_detections - 10}개 ({improvement:+.1f}%)")
    print("   멀리 있는 작은 카트 탐지 개선 확인!")
elif total_detections == 10:
    print(f"\n➖ 동일: 탐지 개수 유지")
else:
    print(f"\n⚠️  감소: {10 - total_detections}개 ({((total_detections - 10) / 10) * 100:.1f}%)")

print("\n" + "=" * 70)
print(f"결과 이미지: {output_dir}/")
print("=" * 70)
