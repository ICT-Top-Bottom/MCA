### 5차 학습

에폭 200(200에서 종료)
배치 24
이미지사이즈 768
이미지 394개
mAP50-95 0.853
yolo11s-seg 모델 사용

```
results = model.train(
    data='data.yaml',
    
    # --- 학습 기본 설정 ---
    epochs=200,
    patience=50,         # 충분한 여유
    batch=24,            # OOM 발생 시 16으로 줄이기
    imgsz=768,           # 640과 1024의 중간 ⭐
    device=[0, 1],
    
    # --- 최적화 ---
    optimizer='auto',    # YOLO11 기본 optimizer (잘 튜닝됨)
    
    # --- 세그멘테이션 특화 ---
    overlap_mask=True,
    close_mosaic=20,     # 마지막 20 epoch는 mosaic 끔 ⭐⭐⭐
    mask_ratio=4,
    
    # --- 데이터 증강 (오탐 방지) ---
    mosaic=1.0,          # 작은 객체 학습에 도움
    mixup=0.0,           # OFF! 카트 겹침 혼동 방지 ⭐⭐⭐
    copy_paste=0.0,      # OFF (혼동 방지)
    
    # --- CCTV 환경 특화 (보수적) ---
    degrees=10.0,        # 15 → 10 (과도한 회전 방지) ⭐
    translate=0.1,       # 적당한 이동
    scale=0.4,           # 0.5 → 0.4 (과도한 크기 변화 방지) ⭐
    perspective=0.0005,  # 최소한의 원근
    flipud=0.0,          # 상하 반전 OFF ⭐
    fliplr=0.5,          # 좌우 반전만 유지
    
    # --- HSV (조명 변화) ---
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    
    # --- Warmup ---
    warmup_epochs=5,
    warmup_momentum=0.8,
    
    # --- 저장 및 로깅 ---
    project='runs/segment',
    name='yolo11s_optimum_v5',
    exist_ok=True,
    verbose=True,
    save=True,
    save_period=10,
    plots=True,
    val=True,
    amp=True
)
```