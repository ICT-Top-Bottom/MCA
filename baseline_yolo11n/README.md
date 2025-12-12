### 1차 학습

에폭 100(99에서 종료)
배치 16
이미지사이즈 512
이미지 397개
mAP50-95 0.822

```
results = model.train(
    data='data.yaml',
    epochs=100,
    batch=16,
    imgsz=512,
    device=[0, 1],  # Multi-GPU
    
    # 기본 설정 (튜닝 없음)
    patience=20,
    
    # 저장 및 로깅
    project='runs/train',
    name='baseline_yolo11n',
    exist_ok=True,
    verbose=True,
    save=True,
    save_period=10,
    plots=True,
    val=True,
    amp=True  # Automatic Mixed Precision
)
```