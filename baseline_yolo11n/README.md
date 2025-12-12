### 1차 학습

에폭 100(99에서 종료)
배치 16
이미지사이즈 512
이미지 397개
mAP50-95 0.822
yolo11n-seg 사용

```
results = model.train(
    data='data.yaml',
    epochs=150,
    batch=24,
    imgsz=640,
    device=[0, 1],  # Multi-GPU
    
    # 조정된 설정
    patience=50,
    
    # 저장 및 로깅
    project='runs/segment',
    name='yolo11n_seg_option1',
    exist_ok=True,
    verbose=True,
    save=True,
    save_period=10,
    plots=True,
    val=True,
    amp=True  # Automatic Mixed Precision
)
```