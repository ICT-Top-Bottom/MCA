### 3차 학습

에폭 150(150에서 종료)
배치 12
이미지사이즈 1024
이미지 394개
mAP50-95 0.725
yolo11n-seg 모델 사용

```
results = model.train(
    data='data.yaml',
    epochs=150,
    batch=12,
    imgsz=1024,
    patience=30,
    device=[0, 1],  # Multi-GPU
    
    # Multi-scale (핵심!)
    multi_scale=True,
    
    # Optimizer: AdamW
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    
    # 증강 설정
    mosaic=1.0,
    mixup=0.15,        # CCTV 조명 변화
    copy_paste=0.0,
    
    # CCTV 각도 증강
    degrees=15.0,      # 회전
    perspective=0.0005, # 원근
    
    # 밝기/대비 강화
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.5,
    
    # 기본 증강
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,
    
    # Warmup
    warmup_epochs=5,
    warmup_momentum=0.8,
    
    # 저장 및 로깅
    project='runs/segment',
    name='yolo11n_seg_cctv_augmented',
    exist_ok=True,
    verbose=True,
    save=True,
    save_period=10,
    plots=True,
    val=True,
    amp=True  # Automatic Mixed Precision
)
```