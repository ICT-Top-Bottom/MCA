### 4차 학습

에폭 200(200에서 종료)
배치 8
이미지사이즈 1280
이미지 394개
mAP50-95 0.688
yolo11s-seg 모델 사용

```
results = model.train(
    data='data.yaml',
    epochs=200,        # 150 → 200 (더 긴 학습)
    batch=8,           # small 모델이라 batch 줄임
    imgsz=1280,        # 1024 → 1280 (작은 객체 탐지)
    patience=40,       # 30 → 40
    device=[0, 1],
    
    # Multi-scale
    multi_scale=True,
    
    # Optimizer
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    
    # 세그멘테이션 특화 ⭐
    overlap_mask=True,   # 겹치는 마스크 학습
    close_mosaic=10,     # 마지막 10 epoch는 mosaic off
    mask_ratio=4,        # 고품질 마스크
    
    # Loss 가중치 (클래스 분류 강화) ⭐
    cls=0.5,             # classification loss (기본 0.5)
    box=7.5,             # box loss (기본 7.5)
    
    # CCTV 증강
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.0,
    degrees=15.0,
    perspective=0.0005,
    
    # HSV
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
    name='yolo11s_seg_option_a',
    exist_ok=True,
    verbose=True,
    save=True,
    save_period=10,
    plots=True,
    val=True,
    amp=True
)
```