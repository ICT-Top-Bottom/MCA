"""
Confidence Threshold 비교 테스트
여러 conf 값으로 테스트해서 최적값 찾기
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
from collections import Counter

# 모델 로드 (학습 완료된 모델로 변경)
MODEL_PATH = 'yolo11s_optimum_v5/best.pt'  # ✅ 확인됨

print("="*70)
print("Confidence Threshold 비교 테스트")
print("="*70)
print(f"모델: {MODEL_PATH}")
print("테스트할 confidence 값: 0.25, 0.30, 0.35, 0.40, 0.45")
print("="*70)

model = YOLO(MODEL_PATH)

# 테스트할 confidence 값들
conf_values = [0.25, 0.30, 0.35, 0.40, 0.45]

# 클래스 정보
class_names = {0: 'combined', 1: 'empty', 2: 'fully'}
class_colors = {
    0: (255, 0, 0),    # combined: 파랑
    1: (0, 0, 255),    # empty: 빨강
    2: (0, 255, 0)     # fully: 초록
}

# 테스트 이미지
test_images = list(Path('testImage').glob('*.jpg')) + list(Path('testImage').glob('*.png'))
test_images = sorted(test_images)

# 결과 저장
results_summary = {}

for conf in conf_values:
    print(f"\\n{'='*70}")
    print(f"Testing with conf={conf}")
    print('='*70)

    output_dir = Path(f'yolo11s_optimum_v5/conf_{int(conf*100)}_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    total_detections = Counter()

    for img_path in test_images:
        # 추론
        results = model.predict(
            str(img_path),
            conf=conf,
            iou=0.45,
            verbose=False,
            retina_masks=True
        )

        # 원본 이미지
        img = cv2.imread(str(img_path))
        annotated = img.copy()
        mask_overlay = img.copy()

        detections = Counter()

        for result in results:
            boxes = result.boxes
            masks = result.masks

            if boxes is not None and len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])

                    color = class_colors[cls]
                    class_name = class_names[cls]

                    # 세그멘테이션 마스크
                    if masks is not None and idx < len(masks):
                        mask = masks[idx].data[0].cpu().numpy()
                        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                        mask_bool = mask_resized > 0.5
                        mask_overlay[mask_bool] = color

                    # 바운딩 박스
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                    # 라벨
                    label = f"{class_name} {confidence:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 5, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    detections[class_name] += 1
                    total_detections[class_name] += 1

        # 마스크 합성
        annotated = cv2.addWeighted(annotated, 0.6, mask_overlay, 0.4, 0)

        # 저장
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), annotated)

    # 결과 출력
    print(f"\\n결과 (conf={conf}):")
    for class_name in ['combined', 'empty', 'fully']:
        count = total_detections[class_name]
        print(f"  {class_name}: {count}개")

    total = sum(total_detections.values())
    print(f"  전체: {total}개")

    results_summary[conf] = {
        'total': total,
        'combined': total_detections['combined'],
        'empty': total_detections['empty'],
        'fully': total_detections['fully']
    }

# 최종 비교
print("\\n" + "="*70)
print("최종 비교 결과")
print("="*70)
print(f"{'Conf':<8} {'Total':<8} {'Combined':<10} {'Empty':<8} {'Fully':<8}")
print("-"*70)
for conf in conf_values:
    r = results_summary[conf]
    print(f"{conf:<8.2f} {r['total']:<8} {r['combined']:<10} {r['empty']:<8} {r['fully']:<8}")

print("\\n" + "="*70)
print("💡 선택 가이드:")
print("  - conf=0.25~0.30: 탐지율 최대 (오탐 가능성 높음)")
print("  - conf=0.35~0.40: 균형잡힌 선택 ⭐ 추천")
print("  - conf=0.45+: 확실한 것만 (놓치는 것 있을 수 있음)")
print("="*70)
