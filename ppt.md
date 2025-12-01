📊 최종 분석 결과 및 전략 결정
🔍 발견된 핵심 문제
1. 신규 데이터의 치명적 불균형
기존 데이터 (1~240): 불균형 1.40:1 (양호)
  - fully: 40.0%
  - empty: 31.4%
  - combined: 28.6%

신규 데이터 (241~): 불균형 53.50:1 (치명적!)
  - fully: 98.2% ← 거의 전부 fully!
  - empty: 0%
  - combined: 1.8%

합산 후: 불균형 1.95:1로 악화
2. 신규 데이터 품질 저하
fully 박스 크기 일관성 저하: 기존 표준편차 0.2251 → 신규 0.0875 (너무 작고 균일함 - 다양성 부족)
combined 데이터: 신규에 단 2개만 존재, 거의 의미 없음
3. 과적합 요인 재확인
yolo11n_649: 250 epoch (과도함)
yolo11s_kaggle: 300 epoch + patience 50 (극도로 과도함)
AdamW optimizer로 인한 빠른 수렴 후 훈련 데이터 암기
💡 최종 결정: "Fully 데이터 제거 + 균형 맞추기"
Augmentation 대신 데이터 제거를 선택한 이유:
✅ 신규 데이터의 품질이 낮음
신규 fully 데이터는 다양성이 부족 (표준편차 낮음)
Augmentation으로 empty/combined를 늘려봤자, 실제 fully 과검출 문제는 해결 안 됨
✅ 기존 240개 데이터가 검증됨
yolo11n_240 모델의 테스트 성능이 우수함을 확인
기존 데이터의 균형(1.40:1)이 훨씬 건강함
✅ "More data ≠ Better"의 교훈
저품질 데이터 409개를 추가한 결과 오히려 성능 저하
고품질 240개 < 저품질 649개
✅ 학습 시간 및 비용 효율성
균형 잡힌 적은 데이터로 빠르게 학습 (epoch 줄임)
Augmentation으로 데이터 늘리면 학습 시간만 증가
🎯 구체적 실행 계획
Phase 1: 데이터 정제 (균형 맞추기)
목표 비율: fully:empty:combined = 35:35:30 현재 상태:
fully: 349파일
empty: 187파일
combined: 141파일
정제 후:
fully: 187파일 (empty와 맞춤, 162개 제거)
empty: 187파일 (유지)
combined: 160파일 (augmentation으로 19개 추가)
제거 전략:
신규 fully 데이터(241~) 107개 전부 제거
기존 fully 데이터 중 박스 크기가 너무 작거나 큰 이상치 55개 추가 제거
Combined는 offline augmentation (회전, 밝기 조정)으로 19개 증강
Phase 2: 학습 설정 최적화
model = YOLO('yolo11n.pt')  # 또는 yolo11s.pt

results = model.train(
    data='data.yaml',
    epochs=80,              # 100 → 80으로 감소
    batch=16,
    imgsz=640,
    
    # 기본 SGD 사용 (AdamW 제거)
    # optimizer 파라미터 삭제
    
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    
    patience=15,            # 30 → 15로 감소 (조기 종료)
    
    device=0,
    amp=True,               # Mixed Precision 추가
    
    # 클래스 가중치 조정 (fully 과검출 억제)
    cls=2.0,                # 분류 loss 가중치 증가
    
    # 기본 augmentation (과도한 augmentation 제거)
    mosaic=1.0,
    mixup=0.0,              # 0.3 → 0.0
    degrees=10.0,           # 15 → 10
    
    save=True,
    plots=True,
    val=True
)
📈 예상 결과
불균형 해소: 1.95:1 → 1.17:1 (건강한 수준)
과적합 방지: 짧은 epoch + patience로 일반화 능력 향상
클래스별 정확도 개선: fully 과검출 감소, empty/combined 재현율 향상
학습 시간 단축: 534개 이미지, 80 epoch → 약 40% 시간 절약
다음 단계로 데이터 정제 스크립트를 만들어드릴까요?