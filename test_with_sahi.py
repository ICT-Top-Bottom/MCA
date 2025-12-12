"""
SAHI (Slicing Aided Hyper Inference) + High Confidence 테스트
오탐 최소화 + 원거리 객체 탐지 최적화
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path
import cv2
from collections import Counter
import numpy as np

print("="*70)
print("SAHI + High Confidence 테스트")
print("="*70)
print("전략:")
print("  1. SAHI로 이미지를 슬라이스해서 추론 (원거리 객체 탐지)")
print("  2. conf=0.40 (오탐 제거)")
print("  3. 결과 비교: 기본 추론 vs SAHI 추론")
print("="*70)

# 모델 경로 (학습 완료된 모델로 변경)
MODEL_PATH = 'yolo11s_optimum_v5/best.pt'  # 학습 완료 후 경로 확인

# SAHI 모델 로드
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',  # YOLO11도 yolov8 타입으로 동작
    model_path=MODEL_PATH,
    confidence_threshold=0.40,  # 높은 threshold ⭐
    device='cuda:0'
)

# 클래스 정보
class_names = {0: 'combined', 1: 'empty', 2: 'fully'}
class_colors = {
    0: (255, 0, 0),    # combined: 파랑
    1: (0, 0, 255),    # empty: 빨강
    2: (0, 255, 0)     # fully: 초록
}

# 테스트 이미지
test_images = Path('testImage')
output_dir = Path('yolo11s_optimum_v5/sahi_test_results')
output_dir.mkdir(parents=True, exist_ok=True)

# 통계
sahi_detections = Counter()
image_count = 0

print("\\n🔍 SAHI 추론 시작...\\n")

for img_path in sorted(list(test_images.glob('*.jpg')) + list(test_images.glob('*.png'))):
    print(f"처리 중: {img_path.name}")

    # SAHI 슬라이싱 추론
    result = get_sliced_prediction(
        str(img_path),
        detection_model,
        slice_height=512,      # 슬라이스 크기 (작을수록 정밀)
        slice_width=512,
        overlap_height_ratio=0.2,  # 겹침 비율
        overlap_width_ratio=0.2,
        verbose=0
    )

    # 원본 이미지 로드
    img = cv2.imread(str(img_path))
    annotated = img.copy()

    detections = Counter()

    # SAHI 결과 그리기
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)

        cls = pred.category.id
        conf = pred.score.value

        color = class_colors[cls]
        class_name = class_names[cls]

        # 바운딩 박스
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # 라벨
        label = f"{class_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 통계
        detections[class_name] += 1
        sahi_detections[class_name] += 1

    # 저장
    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), annotated)

    print(f"  → SAHI: {dict(detections)}")
    image_count += 1

print("\\n" + "="*70)
print("SAHI 테스트 완료!")
print("="*70)
print(f"처리된 이미지: {image_count}개")
print(f"\\nSAHI 탐지 결과 (conf=0.40):")
for class_name in ['combined', 'empty', 'fully']:
    count = sahi_detections[class_name]
    print(f"  {class_name}: {count}개")

total_sahi = sum(sahi_detections.values())
print(f"\\n전체 탐지 객체 (SAHI): {total_sahi}개")

print(f"\\n결과 저장 위치: {output_dir}/")
print("="*70)

print("\\n📊 비교 예상:")
print("  - 기본 추론 (conf=0.25): 오탐 많음")
print("  - SAHI (conf=0.40): 오탐 감소 + 원거리 객체 유지")
print("\\n💡 SAHI 장점:")
print("  ✅ 큰 이미지를 작게 쪼개서 추론 → 원거리 객체도 크게 보임")
print("  ✅ 높은 confidence로 오탐 제거")
print("  ✅ 학습 해상도보다 큰 이미지도 잘 처리")
