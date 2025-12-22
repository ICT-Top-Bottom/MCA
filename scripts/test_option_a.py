"""
옵션 A: v1 + Loss 가중치 조정 모델 테스트
box=10.0, dfl=2.0
"""
from ultralytics import YOLO
from pathlib import Path
import cv2

print("=" * 70)
print("옵션 A: v1 + Loss 가중치 조정 테스트")
print("=" * 70)
print("변경 사항:")
print("  - Box Loss: 7.5 → 10.0 (박스 정확도 강화)")
print("  - DFL Loss: 1.5 → 2.0 (분포 손실 강화)")
print("  - 다른 설정: v1과 동일")
print("=" * 70)

# 옵션 A 모델 로드
print("\n옵션 A 모델 로딩...")
model = YOLO('option_a_loss_weights/best.pt')
print("[OK] 모델 로드 완료")

# 테스트 이미지
test_dir = Path('testImage')
output_dir = Path('test_option_a_results')
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

    # 예측 (기본 파라미터)
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

print("\n비교:")
print("  v1 (기본):         24개 탐지")
print("  옵션 B (Soft-NMS): 21개 탐지 (-3)")
print(f"  옵션 A (Loss):     {total_count}개 탐지")

diff = total_count - 24
if diff > 0:
    print(f"\n✅ v1보다 {diff}개 더 탐지! (박스 정확도 향상 효과)")
    if total_detections['combined'] >= 2:
        print(f"✅ Combined도 {total_detections['combined']}개 탐지 성공!")
elif diff == 0:
    print(f"\n⚠️  v1과 동일 (박스 정확도는 향상되었을 수 있음)")
else:
    print(f"\n❌ v1보다 {abs(diff)}개 감소")

print("\n🔍 박스 정확도가 향상되었는지 이미지 확인 필요!")
