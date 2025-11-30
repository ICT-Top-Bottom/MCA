# MCA 프로젝트 발표 PPT 구성

---

## #1 표지

**MCA: Mart Cart Analysis**
쇼핑카트 상태 감지 시스템

팀: ICT Top-Bottom
YOLO11 Object Detection

---

## #2 프로젝트 개요

**목표**: 쇼핑카트 상태를 실시간 감지 및 분류

| 클래스 | 설명 |
|--------|------|
| fully_cart | 물건이 가득 찬 카트 |
| empty_cart | 빈 카트 |
| combined_cart | 부분적으로 물건이 있는 카트 |

**활용**: 마트 내 카트 회수 자동화, 재고 현황 파악

---

## #3 데이터셋 구축 - 수량 변화

### 1차 → 2차 데이터 수집

| 클래스 | 1차 | 2차 | 증가량 |
|--------|-----|-----|--------|
| empty_cart | 80장 | 196장 | +145% |
| fully_cart | 80장 | 258장 | +222% |
| combined_cart | 80장 | 131장 | +64% |
| **합계** | **240장** | **585장** | **+144%** |

**수집 전략**: 1차 테스트 후 empty_cart, fully_cart의 인식률이 낮아 해당 클래스 데이터 집중 수집

---

## #4 데이터셋 구축 - 촬영 다양성

[실제 촬영 이미지 4~6장 배치]

**촬영 조건 다양화**:
- **조명**: 주차장마다 조명 환경이 상이 → 밝은/어두운 환경 모두 촬영
- **거리**: 근거리(1m) ~ 원거리(5m) 다양한 거리에서 촬영
- **각도**: 정면, 측면, 대각선 등 다양한 각도
- **배경**: 주차장, 매장 내부, 카트 보관소 등

---

## #5 레이블링 방식

**사용 도구**: labelImg (수동 라벨링)

**Roboflow 미사용 이유**:
- 카트가 격자 구조(철망) → 자동 라벨링 정확도 낮음
- 카트 내부와 외부 경계 인식 어려움
- 결국 수동으로 하나하나 직접 라벨링

[labelImg 작업 스크린샷]

**라벨링 포맷**: YOLO 형식 (.txt)
```
<class_id> <x_center> <y_center> <width> <height>
```

---

## #6 학습/검증 데이터 분리

| 구분 | 1차 테스트 | 2차 테스트 |
|------|-----------|-----------|
| 비율 | 8:2 | **7:3** |
| Train | 192장 | 410장 |
| Val | 48장 | 175장 |

**7:3 선택 이유**: 
- 데이터 증가로 검증 세트 확대 가능
- 과적합 방지 및 일반화 성능 검증 강화

---

## #7 클래스 균형성

### 1차 vs 2차 클래스 분포

**1차 (균등 분포)**:
- 각 클래스 80장씩 (33.3% : 33.3% : 33.3%)

**2차 (의도적 불균형)**:
| 클래스 | 수량 | 비율 |
|--------|------|------|
| fully_cart | 258장 | 44% |
| empty_cart | 196장 | 34% |
| combined_cart | 131장 | 22% |

**이유**: 1차 테스트에서 empty_cart→background 오분류 많음 → 해당 클래스 데이터 보강

---

## #8 모델 선택

### YOLO11 선택 이유

| 버전 | 특징 |
|------|------|
| YOLOv8 | 안정적, 널리 사용 |
| **YOLO11** | 최신, 향상된 정확도 |

**YOLO11 장점**:
- 카트 내부 물품 같은 세부 객체 인식 성능 향상
- Detection + Classification 동시 수행

### 모델 크기 변경: n → s

| 구분 | 1차 | 2차 |
|------|-----|-----|
| 모델 | yolo11**n** | yolo11**s** |
| 파라미터 | 2.6M | 9.4M |
| 데이터 | 240장 | 585장 |

**변경 이유**: 데이터 증가 → 더 큰 모델 학습 가능

---

## #9 하이퍼파라미터 설정

| 파라미터 | 1차 | 2차 | 변경 이유 |
|----------|-----|-----|-----------|
| epochs | 100 | 100 | 유지 |
| batch | 16 | 16 | 유지 |
| imgsz | 640 | 640 | 유지 |
| optimizer | SGD | **AdamW** | 수렴 안정성 |
| lr0 | 0.01 | 0.01 | 유지 |
| patience | 20 | **30** | 조기종료 완화 |
| train/val | 8:2 | **7:3** | 검증 강화 |

**Augmentation 강화**:
- mosaic: 1.0 유지
- mixup: 0 → **0.2** 추가
- degrees: 0 → **15.0** (회전)

---

## #10 1차 테스트 결과 (yolo11n, 240장)

### 학습 곡선

[(1차) results.png]

### Confusion Matrix

[(1차) confusion_matrix.png]

**문제점 발견**:
- empty_cart → background 오분류 7건 (36.8%)
- mAP50-95: 0.50 (개선 필요)

---

## #11 2차 테스트 결과 (yolo11s, 585장)

### 학습 곡선

[(2차) results.png]

### Confusion Matrix

[(2차) confusion_matrix.png]

**개선점**: (실제 학습 후 결과 기입)
- empty_cart 오분류 감소
- mAP50-95 향상

---

## #12 성능 비교 (1차 vs 2차)

[(1차) results.png] [(2차) results.png]

### 정량적 비교

| Metric | 1차 (yolo11n) | 2차 (yolo11s) | 변화 |
|--------|--------------|--------------|------|
| mAP50 | 0.80 | (결과) | (+X%) |
| mAP50-95 | 0.50 | (결과) | (+X%) |
| Precision | 0.85 | (결과) | (+X%) |
| Recall | 0.75 | (결과) | (+X%) |

---

## #13 성능 향상 원인 분석

### 향상 요인

| 요인 | 기여도 | 설명 |
|------|--------|------|
| **데이터 증가** | ★★★ | 240장 → 585장 (+144%) |
| **모델 크기** | ★★☆ | nano → small (파라미터 3.6배) |
| **클래스 보강** | ★★☆ | 오분류 많은 클래스 집중 수집 |
| **Augmentation** | ★☆☆ | mixup, rotation 추가 |

### 오류 분석 (1차)

| 오분류 패턴 | 원인 | 해결 |
|------------|------|------|
| empty_cart → background | 빈 카트가 배경과 유사 | 다양한 배경에서 촬영 |
| fully_cart → empty_cart | 물품이 적을 때 혼동 | fully_cart 데이터 확대 |

---

## #14 실시간 추론 시연

### 이미지 추론 결과

[추론 결과 이미지 3장]

### 동영상 추론 (예정)

- 실제 마트 환경에서 촬영한 동영상 테스트
- 실시간 카트 상태 감지 시연

**추론 속도**: ~11ms/image (T4 GPU 기준)

---

## #15 실행 환경

| 항목 | 사양 |
|------|------|
| 플랫폼 | Google Colab |
| GPU | Tesla T4 (15GB) |
| Python | 3.10+ |
| Framework | Ultralytics 8.3+ |
| CUDA | 12.x |

**학습 시간**:
- yolo11n (100 epochs): ~15분
- yolo11s (100 epochs): ~30분

---

## #16 향후 계획

1. **데이터 추가 수집**: 클래스당 150장 이상 목표
2. **모델 고도화**: yolo11m 테스트
3. **실시간 시스템 구축**: 웹캠 연동 실시간 감지
4. **성능 최적화**: TensorRT 변환으로 추론 속도 향상

---

## #17 Q&A

**GitHub**: https://github.com/ICT-Top-Bottom/MCA

**팀원**:
- (팀원 이름들)

감사합니다.

---

# 이미지 배치 가이드

## 필요한 이미지 목록

1. **표지**: 쇼핑카트 이미지 또는 로고
2. **촬영 다양성 (슬라이드 4)**: 다양한 조건에서 촬영한 카트 이미지 4~6장
3. **레이블링 (슬라이드 5)**: labelImg 작업 화면 스크린샷
4. **1차 결과 (슬라이드 10)**:
   - yolo11n_240/results/results.png
   - yolo11n_240/results/confusion_matrix.png
5. **2차 결과 (슬라이드 11)**:
   - yolo11s_xxx/train/results.png
   - yolo11s_xxx/train/confusion_matrix.png
6. **비교 (슬라이드 12)**: 1차/2차 results.png 나란히 배치
7. **추론 결과 (슬라이드 14)**: testImageResult 폴더 이미지

## 비교 그래프 생성 코드 (Colab에서 실행)

```python
import matplotlib.pyplot as plt
import numpy as np

# 1차 vs 2차 비교 막대 그래프
metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
first = [0.80, 0.50, 0.85, 0.75]  # 1차 결과
second = [0.XX, 0.XX, 0.XX, 0.XX]  # 2차 결과 입력

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, first, width, label='1차 (yolo11n)', color='#3498db')
bars2 = ax.bar(x + width/2, second, width, label='2차 (yolo11s)', color='#2ecc71')

ax.set_ylabel('Score')
ax.set_title('1차 vs 2차 성능 비교')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150)
plt.show()
```
