# 호텔 예약 취소 예측 프로젝트

## 프로젝트 개요
호텔 예약 데이터를 기반으로 고객의 예약 취소 여부를 예측하는 머신러닝 모델을 개발했습니다. 다양한 고객 및 예약 정보를 분석하여 취소 가능성을 사전에 예측하고, 호텔 운영 효율성을 높이는 것을 목표로 합니다.

---

## 사용 데이터
* 데이터 출처: [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
* 개수: 119,000+ 개
* 중요하다고 생각하는 컬럼:
  * `is_canceled`: 예약 취소여부 (0: 예약, 1: 취소)
  * `adr`: 한 날 할당 요금
  * `deposit_type`: 보조금 형태 (No Deposit: 보증금 없음, Non Refund: 환불불가, Refundable: 환불가능)
  * `lead_time` : 예약한날로 부터 기다린 시간간
---

## 사용 기술
* Python, Pandas, Scikit-Learn, XGBoost, Matplotlib, etc.

## 전체 전처리 과정
### 1. 결차치 처리
* `company`, `agent`: 결측치가 많고 예측 성능에 큰 영향을 미치지 않아 제거
* `children` 객체를 0으로 대체 후 `int64`로 변환
* `country`: 포르투갈 비중이 40% 이상으로 편중되어 있어 제거

### 2. 이상치 제거
* `adr` > 4000 이상 값 제거
* `babies` 어른없이 아기만 예약한 경우는 제거
* `adults` 값이 5이상인 단체 손님이라 판단되지만 adr이 0으로 표기되어 해당 값 삭제
* `children` 어른 없이 어린이들만 예약한 경우는 제거거 

### 3. 날짜 컬럼 통합
* `arrival_date_year`, `arrival_date_month`, `arrival_date_day_of_month` → `arrival_date` 을 datetime 형식으로 통합 후 기존 컬럼 삭제

### 4. 인원수 통합
* `adults`, `children`, `babies` → `total_guests` adr변동이 없어 한개의 컬럼으로 통합 기존 컬럼 삭제

### 5. 채널 통합
* `market_segment` + `distribution_channel` → 일괄성이 있는 값으로 통일 (예: 'Travel Agency', 'Corporate', 'Direct')

---

## 피처 생성 (Feature Engineering)

### 서비스 이용 점수 (`total_service`)
서비스 이용이 많을 수록 충성고객으로 판단되어 예약 취소율이 낮을 것이라 생각하여 변수 생성
해당 변수는 상관계수를 이용하여 가중치를 지정하여 점수 할당

```python
# 예시 코드
from sklearn.preprocessing import StandardScaler

# 1. 서비스 컬럼 지정
service_cols = ['meal_numeric', 'required_car_parking_spaces', 'total_of_special_requests']

# 2. 상관계수 기반 가중치 직접 지정 (이미 계산된 값 사용)
weights = {
    'meal_numeric': 0.009,  
    'required_car_parking_spaces': 0.448,
    'total_of_special_requests': 0.543
}

# 3. 표준화 (StandardScaler)
scaler = StandardScaler()
scaled = scaler.fit_transform(df_hotel[service_cols])

# 4. 표준화된 값을 데이터프레임으로 변환
scaled_df = pd.DataFrame(scaled, columns=['meal_w', 'parking_w', 'requests_w'])

# 5. 가중치 곱하고 total_service 점수 생성
df_hotel['total_service'] = (
    weights['meal_numeric'] * scaled_df['meal_w'] +
    weights['required_car_parking_spaces'] * scaled_df['parking_w'] +
    weights['total_of_special_requests'] * scaled_df['requests_w']
)
df_hotel.drop(columns=['meal_numeric','required_car_parking_spaces','total_of_special_requests'], inplace=True)
```
---


##  모델링 및 평가

총 4개의 분류 모델을 학습 및 비교

- Logistic Regression 
- Random Forest
- XGBoost 
- LightGBM 

모든 모델에 대해 70:20:10 (Train:Val:Test) 비율로 데이터를 분할하고, Stratified Sampling을 적용했습니다.  
모델 평가는 **불균형 데이터** 특성을 고려하여 **Recall**과 **F1-score**를 중심으로 확인했습니다.

| Model          | Val Recall | Val F1 | Test Recall | Test F1 |
|----------------|------------|--------|--------------|---------|
| Logistic       | 0.72       | 0.72   | 0.73         | 0.72    |
| Random Forest  | 0.83       | 0.83   | 0.83         | 0.84    |
| XGBoost        | 0.82       | 0.82   | 0.83         | 0.82    |
| LightGBM       | 0.81       | 0.81   | 0.82         | 0.82    |


## 평가지표 선정 이유

호텔 예약 취소 예측 문제는 **이진 불균형 분류 문제**로, 전체 예약 중 취소는 소수입니다.

- 단순 정확도(Accuracy)는 항상 "예약됨"으로만 예측해도 높게 나올 수 있어 신뢰하기 어렵습니다.
- 호텔 운영 측면에서는 **실제로 취소할 고객을 놓치지 않는 것**이 중요하기 때문에,
  - **Recall (재현율)**: 실제 취소를 얼마나 잘 맞췄는지를 확인
  - **F1-score**: Precision과 Recall의 균형을 보조적으로 평가

이를 종합해 **Recall 중심의 평가 지표를 중점적으로 활용**했습니다.