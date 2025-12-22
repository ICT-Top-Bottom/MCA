# YOLO11 기반 쇼핑 카트 상태 분류 시스템

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF?style=flat-square&logo=yolo&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-6706CE?style=flat-square&logo=roboflow&logoColor=white)

> 한양대학교 ERICA 스마트융합공학부 스마트ICT융합전공
> CCTV 환경에서의 실시간 쇼핑 카트 탐지 및 세그멘테이션

## 프로젝트 소개

본 프로젝트는 YOLO11 딥러닝 모델을 활용하여 CCTV 환경에서 쇼핑 카트의 상태(fully/empty/combined)를 자동으로 탐지하고 분류하는 실시간 시스템입니다. Instance Segmentation 기술을 통해 카트의 정밀한 윤곽선 정보까지 제공하며, 고해상도 처리로 원거리 객체 탐지 능력을 향상시켰습니다.

### 주요 성과

- **Baseline 모델 (YOLO11n)**: 394장 데이터셋으로 mAP50-95 0.822 달성
- **최종 모델 (YOLO11s-seg)**: Precision 4.7%, Recall 3.8% 향상
- **Segmentation 성능**: mask_mAP50 0.979로 정밀한 윤곽선 추출
- **실전 검증**: CCTV 테스트 영상에서 Baseline 대비 원거리 카트 탐지 능력 향상

## 프로젝트 웹페이지

프로젝트 소개 및 데모 영상은 GitHub Pages를 통해 확인하실 수 있습니다:

**🔗 [https://ict-top-bottom.github.io/MCA/](https://ict-top-bottom.github.io/MCA/)**

## 관련 저장소

- **메인 프로젝트**: [https://github.com/ICT-Top-Bottom/MCA](https://github.com/ICT-Top-Bottom/MCA)
- **Appendix**: [https://github.com/ICT-Top-Bottom/MCA-Appendix](https://github.com/ICT-Top-Bottom/MCA-Appendix)

## 기술 스택

### 모델 학습
- **Framework**: PyTorch, Ultralytics YOLO11
- **Platform**: Kaggle (Dual T4 GPU)
- **Augmentation**: Multi-Scale Training, Rotation, HSV, Mosaic, Mixup

### 데이터셋
- **라벨링 도구**: Roboflow (Instance Segmentation)
- **데이터 크기**: 394장 (Train 316장, Valid 78장)
- **클래스**: fully, empty, combined (총 439개 객체)

### 추론 및 배포
- **언어**: Python 3.8+
- **라이브러리**: OpenCV, NumPy, Ultralytics

## 팀 구성

| 이름 | 역할 |
|------|------|
| **윤태웅** | 모델 학습 및 실험 설계 |
| **박재형** | 데이터 수집 및 라벨링 |

## 논문

연구 논문은 `assets` 폴더에서 확인하실 수 있습니다.

## 문의

프로젝트에 대한 문의사항은 아래 이메일로 연락주세요:

**윤태웅** - taewoong25@hanyang.ac.kr
