# MCA

## 폴더 구조

```
MCA/
  ├── FirstTest/                     # 1차 테스트 결과물
  │   ├── data.yaml
  │   ├── inference.py
  │   ├── inference_advanced.py
  │   ├── train_yolo11n.py
  │   ├── train/
  │   ├── val/
  │   ├── results/
  │   ├── testImageResult/
  │   └── testImageAdvancedResult/
  ├── images/                        # 원본 이미지 (재사용)
  ├── labels/                        # 원본 라벨 (재사용)
  ├── testImage/                     # 테스트 이미지 (재사용)
  └── README.md
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