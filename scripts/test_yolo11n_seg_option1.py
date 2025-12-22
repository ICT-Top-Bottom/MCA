"""
YOLO11n-seg Option 1 모델 추론 테스트
testImage 폴더의 이미지로 테스트하고 결과를 yolo11n_seg_option1/results에 저장
"""
from ultralytics import YOLO
from pathlib import Path
import shutil

# 기존 results 폴더 처리
results_dir = Path('yolo11n_seg_option1/results')
if results_dir.exists():
    try:
        shutil.rmtree(results_dir)
    except PermissionError:
        print(f"Warning: Could not delete {results_dir}, will overwrite existing files")
        pass

# 모델 로드
model_path = 'yolo11n_seg_option1/best.pt'
model = YOLO(model_path)

print("="*60)
print("YOLO11n-seg Option 1 Inference Test")
print("="*60)
print(f"Model: {model_path}")
print(f"Test Images: testImage/")

# testImage 폴더의 모든 이미지
test_images = Path('testImage')
image_files = list(test_images.glob('*.png')) + list(test_images.glob('*.jpg'))

print(f"Found {len(image_files)} test images")
print("")

# 추론 실행
results = model.predict(
    source=str(test_images),
    conf=0.25,
    iou=0.45,
    save=True,
    save_txt=False,
    save_conf=True,
    project='yolo11n_seg_option1',
    name='results',
    exist_ok=True,
    show_boxes=True,  # 바운딩 박스 표시
    show_labels=True,  # 라벨 표시
    retina_masks=True  # 고품질 마스크
)

print(f"\n✅ Inference completed!")
print(f"Results saved to: yolo11n_seg_option1/results/")

# 결과 통계
total_detections = {'combined': 0, 'empty': 0, 'fully': 0}
class_names = {0: 'combined', 1: 'empty', 2: 'fully'}

for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            if cls in class_names:
                total_detections[class_names[cls]] += 1

print("\n" + "="*60)
print("Detection Statistics")
print("="*60)
print(f"  Combined: {total_detections['combined']}")
print(f"  Empty:    {total_detections['empty']}")
print(f"  Fully:    {total_detections['fully']}")
print(f"  Total:    {sum(total_detections.values())}")
print("="*60)
