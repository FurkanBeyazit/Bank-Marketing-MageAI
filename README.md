# Bank Marketing Data Analysis (은행 마케팅 데이터 분석)

## Overview (개요)

In this project, we address the problem of imbalanced data (불균형 데이터) in a bank marketing dataset. Our primary objective was to improve the recall score (리콜 점수) by applying various techniques, including feature engineering (피처 엔지니어링) and using different approaches to handle class imbalance (클래스 불균형).

## Key Techniques (주요 기술)

1. **Feature Engineering (피처 엔지니어링):**
   - We transformed and engineered new features (새로운 피처를 변환하고 생성) to improve model performance (모델 성능을 향상시키기 위해).
2. **Random Forest Classifier (랜덤 포레스트 분류기):**
   - A powerful ensemble method (강력한 앙상블 방법) used as our primary model (주 모델로 사용).
3. **Handling Imbalanced Data (불균형 데이터 처리):**
   - **Downsampling (다운샘플링):** Reduced the number of instances (인스턴스 수를 줄임) from the majority class (다수 클래스의).
   - **SMOTE (Synthetic Minority Over-sampling Technique):** Generated synthetic samples (합성 샘플을 생성) for the minority class (소수 클래스에 대해).
   - **Class Weights (클래스 가중치):** Adjusted class weights (클래스 가중치를 조정) to balance the importance of each class (각 클래스의 중요성을 균형 있게 맞춤).

## Model Tuning and Validation (모델 튜닝 및 검증)

We performed model fine-tuning (모델 파인튜닝) using cross-validation (교차 검증) to find the best hyperparameters (최적의 하이퍼파라미터). Our primary evaluation metric (주요 평가 지표) was the recall score (리콜 점수), which we aimed to maximize (최대화하는 것이 목표).

## Results (결과)

After applying these techniques (이러한 기술을 적용한 후), we observed no significant improvements in the recall score (리콜 점수에서). The final model's parameters and performance metrics (최종 모델의 매개변수 및 성능 지표) are detailed in the results section (결과 섹션에 자세히 설명).

## Loaders (로드러)

The data loading (데이터 로드) process involves extracting (추출), transforming (변환), and loading (로딩) data from various sources. We implemented custom loaders to handle different data formats and structures efficiently.

## Transformers (변환기)

Our data transformation (데이터 변환) pipeline includes:
- **Data Cleaning (데이터 정리):** Removing inconsistencies and handling missing values.
- **Feature Scaling (피처 스케일링):** Standardizing features to improve model convergence.
- **Encoding (인코딩):** Converting categorical variables into numerical formats.

## Custom ML Components (커스텀 ML 구성 요소)

We developed custom machine learning components (커스텀 머신러닝 구성 요소) tailored to our specific needs:
- **Custom Feature Selectors (커스텀 피처 선택기):** To select the most relevant features for our models.
- **Custom Evaluation Metrics (커스텀 평가 지표):** Beyond standard metrics, we included specific measures to evaluate model performance in terms of recall and precision.

## MAGEAI ETL Code Integration (MAGEAI ETL 코드 통합)

This project also includes integration with MAGEAI's ETL (Extract, Transform, Load) processes. MAGEAI's ETL framework (MAGEAI의 ETL 프레임워크) was utilized to streamline data processing, ensuring seamless data flow from extraction to model training.

## How to Use (사용 방법)

To replicate the results (결과를 재현하려면), follow the instructions in the `src` folder (src 폴더의 지침을 따르십시오). All necessary scripts, including loaders, transformers, and custom ML components, along with data preprocessing steps (필요한 모든 스크립트와 데이터 전처리 단계), are provided (제공됩니다).

---
