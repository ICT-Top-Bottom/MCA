"""
TTA (Test Time Augmentation) Detection
테스트 시 여러 augmentation 적용 후 voting으로 안정성 향상
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

print("=" * 70)
print("TTA (Test Time Augmentation) Detection")
print("=" * 70)
print("\nStrategy:")
print("  1. 원본 이미지")
print("  2. 좌우 반전")
print("  3. 밝기 조정 (+30%, -30%)")
print("  4. 대비 조정")
print("  5. 약간 스케일 변화")
print("  → 모든 결과를 voting하여 최종 예측")
print("=" * 70)

# 모델 로드
print("\n모델 로딩...")
model = YOLO('yolo11n_balanced_v2/best.pt')
print("[OK] 모델 로드 완료")

# 테스트 이미지
test_dir = Path('testImage')
output_dir = Path('test_tta_results')
output_dir.mkdir(exist_ok=True)

test_images = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
print(f"\n총 {len(test_images)}개 이미지 테스트")

# 클래스 이름
class_names = {0: 'fully', 1: 'empty', 2: 'combined'}

# 통계
total_detections = {'fully': 0, 'empty': 0, 'combined': 0}
total_count = 0

def apply_brightness(img, factor):
    """밝기 조정"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_contrast(img, factor):
    """대비 조정"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = lab[:, :, 0] * factor
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def get_augmentations(img):
    """다양한 augmentation 생성"""
    h, w = img.shape[:2]
    augs = []

    # 1. 원본
    augs.append(('original', img, lambda boxes: boxes))

    # 2. 좌우 반전
    flipped = cv2.flip(img, 1)
    def flip_boxes(boxes):
        for box in boxes:
            x1, y1, x2, y2 = box['xyxy']
            box['xyxy'] = np.array([w - x2, y1, w - x1, y2])
        return boxes
    augs.append(('flip', flipped, flip_boxes))

    # 3. 밝기 증가
    bright_up = apply_brightness(img, 1.3)
    augs.append(('bright+', bright_up, lambda boxes: boxes))

    # 4. 밝기 감소
    bright_down = apply_brightness(img, 0.7)
    augs.append(('bright-', bright_down, lambda boxes: boxes))

    # 5. 대비 증가
    contrast_up = apply_contrast(img, 1.2)
    augs.append(('contrast+', contrast_up, lambda boxes: boxes))

    # 6. 약간 축소
    scale = 0.9
    scaled_down = cv2.resize(img, None, fx=scale, fy=scale)
    padded = cv2.copyMakeBorder(
        scaled_down,
        int((h - scaled_down.shape[0]) / 2),
        h - scaled_down.shape[0] - int((h - scaled_down.shape[0]) / 2),
        int((w - scaled_down.shape[1]) / 2),
        w - scaled_down.shape[1] - int((w - scaled_down.shape[1]) / 2),
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )
    def scale_down_boxes(boxes):
        offset_y = int((h - scaled_down.shape[0]) / 2)
        offset_x = int((w - scaled_down.shape[1]) / 2)
        for box in boxes:
            x1, y1, x2, y2 = box['xyxy']
            box['xyxy'] = np.array([
                (x1 - offset_x) / scale,
                (y1 - offset_y) / scale,
                (x2 - offset_x) / scale,
                (y2 - offset_y) / scale
            ])
        return boxes
    augs.append(('scale0.9', padded, scale_down_boxes))

    # 7. 약간 확대
    scale = 1.1
    scaled_up = cv2.resize(img, None, fx=scale, fy=scale)
    cropped = scaled_up[
        int((scaled_up.shape[0] - h) / 2):int((scaled_up.shape[0] - h) / 2) + h,
        int((scaled_up.shape[1] - w) / 2):int((scaled_up.shape[1] - w) / 2) + w
    ]
    def scale_up_boxes(boxes):
        offset_y = int((scaled_up.shape[0] - h) / 2)
        offset_x = int((scaled_up.shape[1] - w) / 2)
        for box in boxes:
            x1, y1, x2, y2 = box['xyxy']
            box['xyxy'] = np.array([
                x1 / scale + offset_x,
                y1 / scale + offset_y,
                x2 / scale + offset_x,
                y2 / scale + offset_y
            ])
        return boxes
    augs.append(('scale1.1', cropped, scale_up_boxes))

    return augs

def iou(box1, box2):
    """IoU 계산"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def tta_voting(all_predictions, iou_threshold=0.5, vote_threshold=3):
    """
    TTA voting

    Parameters:
    - vote_threshold: 최소 몇 개의 augmentation에서 탐지되어야 하는지
    """
    if not all_predictions:
        return []

    # 모든 예측 수집
    all_boxes = []
    for aug_name, boxes in all_predictions:
        for box in boxes:
            all_boxes.append(box)

    # Clustering: 비슷한 위치의 박스들을 그룹화
    clusters = []
    for box in all_boxes:
        # 기존 클러스터와 겹치는지 확인
        matched = False
        for cluster in clusters:
            # 클러스터의 대표 박스와 비교
            if iou(box['xyxy'], cluster[0]['xyxy']) > iou_threshold:
                cluster.append(box)
                matched = True
                break

        if not matched:
            clusters.append([box])

    # Voting: 충분한 지지를 받은 클러스터만 선택
    final_boxes = []
    for cluster in clusters:
        if len(cluster) >= vote_threshold:  # 최소 vote_threshold개 이상
            # 클러스터 내에서 평균 계산
            avg_xyxy = np.mean([b['xyxy'] for b in cluster], axis=0)

            # 클래스는 다수결
            classes = [b['cls'] for b in cluster]
            most_common_cls = max(set(classes), key=classes.count)

            # Confidence는 평균
            avg_conf = np.mean([b['conf'] for b in cluster])

            final_boxes.append({
                'xyxy': avg_xyxy,
                'conf': avg_conf,
                'cls': most_common_cls,
                'votes': len(cluster)
            })

    return final_boxes

print("\n" + "=" * 70)
print("TTA 탐지 시작")
print("=" * 70)

for img_path in test_images:
    print(f"\n처리중: {img_path.name}")

    # 이미지 읽기
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARNING] 이미지 로드 실패")
        continue

    # Augmentation 생성
    augmentations = get_augmentations(img)
    print(f"  {len(augmentations)}가지 augmentation 적용")

    # 각 augmentation마다 예측
    all_predictions = []
    for aug_name, aug_img, transform_back in augmentations:
        # 예측
        results = model.predict(aug_img, conf=0.20, iou=0.45, verbose=False)
        pred = results[0]

        # 박스 변환
        boxes = []
        if pred.boxes is not None and len(pred.boxes) > 0:
            for box in pred.boxes:
                boxes.append({
                    'xyxy': box.xyxy[0].cpu().numpy().copy(),
                    'conf': float(box.conf[0]),
                    'cls': int(box.cls[0])
                })

            # 원본 좌표계로 변환
            boxes = transform_back(boxes)

        all_predictions.append((aug_name, boxes))
        print(f"    {aug_name}: {len(boxes)}개")

    # TTA Voting
    final_boxes = tta_voting(all_predictions, vote_threshold=3)
    print(f"  → TTA 최종: {len(final_boxes)}개 탐지")

    # 결과 시각화
    result_img = img.copy()

    for box in final_boxes:
        x1, y1, x2, y2 = map(int, box['xyxy'])
        conf = box['conf']
        cls = box['cls']
        votes = box['votes']

        # 클래스별 색상
        colors = {
            0: (0, 255, 0),    # fully: 초록
            1: (0, 0, 255),    # empty: 빨강
            2: (255, 0, 0)     # combined: 파랑
        }
        color = colors[cls]

        # 바운딩 박스
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        # 라벨 (투표수 표시)
        label = f"{class_names[cls]} {conf:.2f} [{votes}/7]"

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
print("TTA 전략 요약")
print("=" * 70)
print("1. 7가지 augmentation 적용")
print("2. 각각에서 예측 수행 (conf > 0.20)")
print("3. 최소 3/7 이상 투표된 결과만 채택")
print("4. 위치/클래스는 평균/다수결로 결정")
print("=" * 70)
print(f"\n결과 저장: {output_dir}/")
