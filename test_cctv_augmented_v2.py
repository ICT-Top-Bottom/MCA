"""
YOLO11n CCTV Augmented v2 모델 테스트
Combined 카트 + 멀리 있는 카트 + 어두운 조명 개선 버전
"""
from ultralytics import YOLO
from pathlib import Path
import cv2

print("=" * 70)
print("YOLO11n CCTV Augmented v2 모델 테스트")
print("=" * 70)
print("개선 사항:")
print("  - MixUp 제거 (격자 구조 보존)")
print("  - Copy-Paste 추가 (Combined 대응)")
print("  - Scale 0.9 (멀리 있는 카트)")
print("  - hsv_v 0.6 (어두운 조명)")
print("=" * 70)

# 모델 로드
print("\n모델 로딩...")
model = YOLO('yolo11n_cctv_augmented_v2/best.pt')
print("[OK] 모델 로드 완료")

# 테스트 이미지
test_dir = Path('testImage')
output_dir = Path('test_cctv_augmented_v2_results')
output_dir.mkdir(exist_ok=True)

test_images = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
print(f"\n총 {len(test_images)}개 이미지 테스트")

# 클래스 이름
class_names = {0: 'fully', 1: 'empty', 2: 'combined'}

# 통계
total_detections = {'fully': 0, 'empty': 0, 'combined': 0}
total_count = 0

print("\n" + "=" * 70)
print("추론 시작")
print("=" * 70)

for img_path in test_images:
    print(f"\n처리중: {img_path.name}")

    # 이미지 읽기
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARNING] 이미지 로드 실패")
        continue

    # 예측
    results = model.predict(img, conf=0.25, iou=0.45, verbose=False)
    pred = results[0]

    # 결과 시각화
    result_img = img.copy()

    detection_count = 0
    class_count = {'fully': 0, 'empty': 0, 'combined': 0}

    if pred.boxes is not None and len(pred.boxes) > 0:
        for box in pred.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # 클래스별 색상
            colors = {
                0: (0, 255, 0),    # fully: 초록
                1: (0, 0, 255),    # empty: 빨강
                2: (255, 0, 0)     # combined: 파랑
            }
            color = colors[cls]

            # 바운딩 박스
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

            # 라벨
            label = f"{class_names[cls]} {conf:.2f}"

            # 배경
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result_img, (x1, y1 - 20), (x1 + w, y1), color, -1)

            # 텍스트
            cv2.putText(result_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 통계
            total_detections[class_names[cls]] += 1
            class_count[class_names[cls]] += 1
            detection_count += 1

    print(f"  → {detection_count}개 탐지")
    print(f"     fully: {class_count['fully']}, empty: {class_count['empty']}, combined: {class_count['combined']}")
    total_count += detection_count

    # 저장
    cv2.imwrite(str(output_dir / img_path.name), result_img)

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)

print("\n클래스별 탐지 결과:")
for cls_name, count in total_detections.items():
    print(f"  {cls_name}: {count}개")

print(f"\n총 탐지된 객체: {total_count}개")
print(f"\n결과 저장: {output_dir}/")
