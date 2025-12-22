"""
YOLO11 카트 탐지 모델 학습 코드
Google Colab 환경에서 실행
"""

# 1. empty_6.jpg의 빈 라벨 파일 생성
print("empty_6.txt 라벨 파일 생성 중...")
with open('train/labels/empty_6.txt', 'w') as f:
    pass  # 빈 파일 생성 (객체가 없는 이미지)

print("empty_6.txt 라벨 파일 생성 완료!")

# 2. Ultralytics 설치
print("\nUltralytics 설치 중...")
import os
os.system('pip install -q ultralytics')

# 3. YOLO11 학습 시작
from ultralytics import YOLO
import torch

print("\n" + "="*60)
print("YOLO11 Cart Detection Training")
print("="*60)

# GPU 확인
print(f"\nGPU 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ GPU를 사용할 수 없습니다. CPU로 학습합니다.")

# 모델 로드
print("\nYOLO11n 모델 로드 중...")
model = YOLO('yolo11n.pt')

# 학습 설정
print("\n학습 설정:")
print("- 데이터셋: data.yaml")
print("- Epochs: 100")
print("- 이미지 크기: 640x640")
print("- 배치 크기: 16")
print("- Patience: 20 (early stopping)")
print("- AMP: True (Mixed Precision)")

print("\n" + "="*60)
print("학습 시작...")
print("="*60 + "\n")

# 학습 실행
results = model.train(
    data='data.yaml',           # 데이터셋 설정 파일
    epochs=100,                 # 학습 epoch 수
    imgsz=640,                  # 이미지 크기
    batch=16,                   # 배치 크기
    device=0,                   # GPU 장치 번호 (CPU는 'cpu')
    patience=20,                # Early stopping patience
    save=True,                  # 모델 저장
    project='runs/train',       # 프로젝트 폴더
    name='cart_detection',      # 실험 이름
    exist_ok=True,              # 기존 폴더 덮어쓰기 허용
    verbose=True,               # 상세 로그 출력
    amp=True,                   # Mixed Precision 학습
    plots=True,                 # 학습 결과 그래프 생성
)

print("\n" + "="*60)
print("학습 완료!")
print("="*60)
print(f"\n최고 성능 모델: runs/train/cart_detection/weights/best.pt")
print(f"마지막 모델: runs/train/cart_detection/weights/last.pt")
print(f"\n학습 결과 그래프: runs/train/cart_detection/results.png")
print(f"혼동 행렬: runs/train/cart_detection/confusion_matrix.png")