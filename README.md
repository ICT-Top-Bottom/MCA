# MCA

## 폴더 구조

```
MCA/
├── data.yaml                      # YOLO11 데이터셋 설정 파일
├── inference.py                   # 기본 추론 코드 (단일 이미지)
├── inference_advanced.py          # 고급 추론 코드 (TTA 적용)
├── train/                         # 학습 데이터셋
│   ├── images/                   # 192개 학습 이미지
│   └── labels/                   # 191개 학습 라벨
├── val/                           # 검증 데이터셋
│   ├── images/                   # 48개 검증 이미지
│   └── labels/                   # 48개 검증 라벨
├── results/                       # 학습 결과 폴더
│   ├── weights/                  # 학습된 모델 가중치
│   │   ├── best.pt               # 최고 성능 모델
│   │   └── last.pt               # 마지막 epoch 모델
│   ├── results.csv               # 학습 메트릭 데이터
│   ├── results.png               # 학습 결과 그래프
│   ├── confusion_matrix.png
│   ├── BoxPR_curve.png
│   └── ... (기타 평가 이미지)
├── testImage/                     # 테스트 이미지 폴더
├── testImageResult                # 테스트 이미지 추론 결과 폴더
├── testImageAdvancedResult/       # 고급 추론 결과 폴더
├── images/                        # 원본 이미지 (240개)
├── labels/                        # 원본 라벨
```

## Google Colab에서 불러오기

```bash
!git clone https://github.com/ICT-Top-Bottom/MCA.git        # 1. 레포지토리 클론
%cd MCA                                                     # 2. 디렉토리 이동
!ls -la                                                     # 3. 폴더 구조 확인
```

## 참고사항

- **empty_6.jpg**: 라벨 파일 누락됨 (학습 데이터에서 제외 권장)
- **클래스**: fully_cart, empty_cart, combined_cart