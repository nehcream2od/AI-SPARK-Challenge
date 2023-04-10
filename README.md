# AI-SPARK-Challenge

## Overview

AI-SPARK-Challenge는 비지도 학습 방식을 사용하여 산업용 공기 압축기의 이상을 탐지하는 프로젝트입니다. 이 프로젝트에서는 시계열 기반 방법을 사용하지 않고, GAN 및 AnoGAN과 같은 모델을 사용하여 문제를 해결합니다.

## Objectives

1. 산업용 공기압축기의 이상 유무 탐지
2. 비지도 학습 방식을 활용한 이상 유무 탐지 모델 개발
3. AutoEncoder, GAN과 같은 모델을 적용하여 문제 해결

## 구현 과정

1. 데이터 분석 및 전처리: 데이터를 탐색하고 이상치 처리, 정규화 등의 전처리 작업을 수행합니다.
2. 모델 구현: 딥러닝 모델을 사용하여 이상을 탐지하는 모델을 구현합니다.
3. 모델 평가: AI factory 대회 사이트를 통해 모델의 성능을 평가합니다.

## 사용 기술 및 라이브러리

- Python
- PyTorch
- Pytorch Lightning
- GAN (Generative Adversarial Network)

## 프로젝트 구조
```bash
AI-SPARK-Challenge/
│
├── base/ # base 파일들의 디렉토리
│
├── datasets/ # torch dataset을 만드는 파일 디렉토리
│ └── dataset.py # dataset 함수
│
├── data_module/ # dataloader, 전처리 관련 디렉토리
│ ├── data_module.py # dataloader, 전처리 적용 함수 파일
│ └── scaler.py # 스케일러 함수 파일
│
├── model/ # 모델 관련 디렉토리
│ ├── model.py # 모델 구현 파일
│ ├── loss.py # loss function 파일
│ ├── scheduler.py # scheduler 파일
│ └── optimizer.py # 옵티마이저 함수 파일
│
├── trainer/ # torch lightning trainer 디렉토리
│ └── trainer.py # trainer 구현 파일, forward, backward 과정 
│
├── utils/ # 유틸리티 함수 디렉토리
│ └── util.py # 유틸리티 함수 파일 
│
├── util.py # 메인 실행 스크립트
│
└── README.md # 프로젝트 설명 파일
```
## 실행 방법
1. repository를 복사합니다.
```bash
$ git clone https://github.com/nehcream2od/AI-SPARK-Challenge.git
```
2. 실행에 필요한 패키지들을 install 합니다.
```bash
$ pip install -r requirements.txt
```
3. 데이터셋을 data/ 디렉토리에 위치시킵니다.(대회 규정으로 인해 데이터는 제공해드릴 수 없음을 양해 부탁드립니다.)
4. 메인 실행 스크립트를 실행합니다.
```bash
$ python train.py -c config.json
```
