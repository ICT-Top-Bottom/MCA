# MCA

## 폴더 구조

```
MCA/
├── data.yaml           # YOLO11 데이터셋 설정 파일
├── train/              # 학습 데이터셋
│   ├── images/        # 192개 학습 이미지
│   └── labels/        # 191개 학습 라벨
├── val/                # 검증 데이터셋
│   ├── images/        # 48개 검증 이미지
│   └── labels/        # 48개 검증 라벨
├── images/             # 원본 이미지 (240개)
├── labels/             # 원본 라벨
└── tests/              # 테스트 관련 파일
```

## Google Colab에서 불러오기

```
!git clone https://github.com/ICT-Top-Bottom/MCA.git        # 1. 레포지토리 클론
%cd MCA                                                     # 2. 디렉토리 이동
!ls -la                                                     # 3. 폴더 구조 확인
```

### 참고사항
empty_6.jpg의 라벨파일이 없음.