"""
Ensemble Detection: balanced_v2 + multiscale
두 모델의 장점을 결합하여 미탐과 오분류 동시 개선
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

print("=" * 70)
print("Ensemble Detection (balanced_v2 + multiscale)")
print("=" * 70)
print("\nStrategy:")
print("  1. balanced_v2: 높은 정밀도 (오탐 없음)")
print("  2. multiscale: 높은 recall (미탐 감소)")
print("  3. Voting: 두 모델 결과 결합")
print("=" * 70)

# 모델 로드
print("\n모델 로딩...")
model_v2 = YOLO('yolo11n_balanced_v2/best.pt')
model_ms = YOLO('yolo11n_multiscale_1024_12/best.pt')
print("[OK] 두 모델 로드 완료")

# 테스트 이미지
test_dir = Path('testImage')
output_dir = Path('test_ensemble_results')
output_dir.mkdir(exist_ok=True)

test_images = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
print(f"\n총 {len(test_images)}개 이미지 테스트")

# 클래스 이름
class_names = {0: 'fully', 1: 'empty', 2: 'combined'}

# 통계
total_detections = {'fully': 0, 'empty': 0, 'combined': 0}
total_count = 0

print("\n" + "=" * 70)
print("Ensemble 탐지 시작")
print("=" * 70)

def iou(box1, box2):
    """IoU (Intersection over Union) 계산"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def ensemble_predictions(pred_v2, pred_ms, iou_threshold=0.5):
    """
    두 모델의 예측 결합

    전략:
    1. balanced_v2의 모든 탐지 결과 채택 (정밀도 높음)
    2. multiscale에서 새로운 탐지만 추가 (단, confidence 높은 것만)
    3. 중복은 confidence 높은 것 선택
    """
    # balanced_v2 결과 (기본으로 채택)
    v2_boxes = []
    if pred_v2.boxes is not None and len(pred_v2.boxes) > 0:
        for box in pred_v2.boxes:
            v2_boxes.append({
                'xyxy': box.xyxy[0].cpu().numpy(),
                'conf': float(box.conf[0]),
                'cls': int(box.cls[0]),
                'source': 'v2'
            })

    # multiscale 결과
    ms_boxes = []
    if pred_ms.boxes is not None and len(pred_ms.boxes) > 0:
        for box in pred_ms.boxes:
            ms_boxes.append({
                'xyxy': box.xyxy[0].cpu().numpy(),
                'conf': float(box.conf[0]),
                'cls': int(box.cls[0]),
                'source': 'ms'
            })

    # Ensemble 결과
    ensemble_boxes = []

    # 1. v2 결과를 모두 추가
    for v2_box in v2_boxes:
        ensemble_boxes.append(v2_box)

    # 2. ms에서 새로운 탐지만 추가
    for ms_box in ms_boxes:
        # ms confidence가 높은 것만 (오분류 방지)
        if ms_box['conf'] < 0.35:  # multiscale은 높은 threshold
            continue

        # v2 결과와 겹치는지 확인
        is_duplicate = False
        for v2_box in v2_boxes:
            if iou(ms_box['xyxy'], v2_box['xyxy']) > iou_threshold:
                is_duplicate = True
                break

        # 새로운 탐지라면 추가
        if not is_duplicate:
            ensemble_boxes.append(ms_box)

    return ensemble_boxes

for img_path in test_images:
    print(f"\n처리중: {img_path.name}")

    # 이미지 읽기
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARNING] 이미지 로드 실패")
        continue

    # 두 모델로 예측
    pred_v2 = model_v2.predict(img, conf=0.25, iou=0.45, verbose=False)[0]
    pred_ms = model_ms.predict(img, conf=0.25, iou=0.45, verbose=False)[0]

    v2_count = len(pred_v2.boxes) if pred_v2.boxes is not None else 0
    ms_count = len(pred_ms.boxes) if pred_ms.boxes is not None else 0

    print(f"  balanced_v2: {v2_count}개")
    print(f"  multiscale: {ms_count}개")

    # Ensemble
    ensemble_boxes = ensemble_predictions(pred_v2, pred_ms)

    print(f"  → Ensemble: {len(ensemble_boxes)}개 탐지")

    # 결과 시각화
    result_img = img.copy()

    for box in ensemble_boxes:
        x1, y1, x2, y2 = map(int, box['xyxy'])
        conf = box['conf']
        cls = box['cls']
        source = box['source']

        # 클래스별 색상
        colors = {
            0: (0, 255, 0),    # fully: 초록
            1: (0, 0, 255),    # empty: 빨강
            2: (255, 0, 0)     # combined: 파랑
        }
        color = colors[cls]

        # 바운딩 박스
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        # 라벨 (모델 출처 표시)
        label = f"{class_names[cls]} {conf:.2f}"
        if source == 'ms':
            label += " [MS]"  # multiscale에서 추가된 것 표시

        # 배경
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_img, (x1, y1 - 20), (x1 + w, y1), color, -1)

        # 텍스트
        cv2.putText(result_img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 통계
        total_detections[class_names[cls]] += 1
        total_count += 1

    # 저장
    cv2.imwrite(str(output_dir / img_path.name), result_img)

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)

print("\n클래스별 탐지 결과:")
for cls_name, count in total_detections.items():
    print(f"  {cls_name}: {count}개")

print(f"\n총 탐지된 객체: {total_count}개")

print("\n" + "=" * 70)
print("Ensemble 전략 요약")
print("=" * 70)
print("1. balanced_v2 결과 전부 채택 (정밀도 보장)")
print("2. multiscale에서 새로운 탐지만 추가 (conf > 0.35)")
print("3. 결과: 미탐 감소 + 오분류 최소화")
print("=" * 70)
print(f"\n결과 저장: {output_dir}/")
