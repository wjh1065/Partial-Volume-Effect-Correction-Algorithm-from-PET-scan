# Readme markdown 연습

# 소개
기존 PET 영상의 Partial Volume effect Correction (PVC)는 주로 Matlab SPM tool box를 이용하여 진행한다.  
본 연구의 주안점은 오랜 시간이 걸리는 PVC 과정을 딥러닝 알고리즘 (3D-ResUnet)으로 대체하여 정확도를 유지하면서도 처리 시간을 단축하고자 한다.

# 순서: 
1. 데이터 셋 소개
2. 데이터 전처리
3. Partial volume Effect correction (PVC)
4. 딥러닝 모델, 3D-ResUnet
5. 학습 방법
6. 정량적 평가 방법
7. 실험 결과 
8. 결론 및 향후 연구 방향


## 1. 데이터 셋 소개
본 연구에서는 알츠하이머 병 뇌 영상 데이터베이스 (ADNI)로 부터 획득한 것이다.  
ADNI 데이터셋에서 694명의 PET영상을 사용하였으며 (복셀 사이즈: 1 ~ 1.25 x 0.92 ~ 1.25 x 0.923 ~ 1.2 mm3, 영상 크기: 160 ~ 256 x 192 ~ 288 x 160 ~ 288), 이 중 550명의 환자 영상을 훈련 데이터로, 50명 환자 영상을 검증 데이터로, 나머지 94명을 테스트 데이터로 설정하고 학습 및 실험을 진행하였다.

## 2. 데이터 전처리
딥러닝 모델에 넣어주기 위해, 모든 영상의 크기를 일정하게 통일시켜주었다.  
Statistical Parametric Mapping(SPM) 패키지를 이용하여 영상 데이터의 크기를 표준뇌 참조 영상과 같아지도록 리슬라이스(reslice)하여 준비하였다 (표준뇌 영상 크기: 복셀 사이즈 1 x 1 x 1 mm3, 크기 256 x 256 x 256).

## 3. Partial volume Effect correction (PVC)
PVC는 Partial volume effect(PVE)를 줄이고 PET 영상에서의 베타 아밀로이드 단백질의 측정을 더욱 정확하게 한다.  
PVE는 영상의 해상도가 낮아서 발생하며 높게 측정된 값은 다소 작게, 큰 값 주변에서는 값이 조금 상승하는 blur 효과라고 할 수 있다.

## 4. 딥러닝 모델, 3D-ResUnet
본 연구에서는 3D-ResUnet 모델 혹은 Vnet을 패치 기반(patch-based)으로 변형하여 사용하였다.  

-- 이미지 넣기 --  


### 4-1. patch based train
이미지를 여러 개로 쪼갠 패치(patch)를 학습데이터로 사용하였는데, PVC는 이미지의 지역적인 특성(local feature)을 보정하는 것이기 때문에, 이렇게 진행하여도 성능에 문제가 없다.  
3차원 영상은 영상의 데이터가 크기 때문에 그래픽 카드 메모리의 한계로 인하여 보통 학습을 진행하는데 어려움을 겪게 된다.  
패치를 사용하게 되면, 각 이미지의 크기가 줄어들어, 적은 메모리를 가진 하드웨어로도 한 번에 여러 개를 한 번에 학습하는 효과가 있어, 학습이 안정화되고 가속화되는 장점을 얻을 수 있다.

### 4-2. intensity cropping
PET영상은 주입한 방사성 물질의 양, 주입한 시간 등에 따라서 영상의 강도가 매우 달라지는 특성이 있다.  
그러하여, PET 영상의 복셀 값 분포를 0에서 1사이의 값으로 매핑해 주었다(MinMaxScaler).  

-- 이미지 넣기 --  

## 5. 학습 방법
PVC 처리 전 PET 영상 데이터를 입력 데이터로 사용하였고 PVC 처리 후 PET 영상 데이터는 정답 데이터로 사용하여 학습을 진행하였다.  
최소제곱에러(Mean Square Error, MSE, 식 (1))를 손실함수로 사용하여, 입력 데이터와 정답 데이터간의 차이를 줄이도록 하였다.

-- 식  --  

본 연구에서는 위에 설명한대로 패치를 이용하여 미니배치의 크기(mini-batch size)를 늘렸는데, 한 사람의 영상을 8개의 패치(각 영상의 실제 크기: 128 x 128 x 128 voxels)로 나누고, 미니배치의 크기(batch size)를 4로 하였으며, 총 300번의 에폭(epoch) 동안 학습을 진행하였다. 최적화 알고리즘(optimizer)는 Adam을 사용했다.  

모든 구현은 유닉스 상(Ubuntu 18.04.5 LTS)의 Python 버전 3.8.8에서 구글 텐서플로우 (TensorFlowTM) 버전 2.5.0, 케라스(Keras) 버전 2.5.0으로 구현되었다. 학습에 사용된 하드웨어 정보는 다음과 같다 (CPU: Intel i7-6700@3.4GHz, GPU: Nvidia RTX 3090 24GB, memory: 48GB).

## 6. 정량적 평가 방법
모델의 평가 방법은 정답 데이터와의 유사성을 비교하는 구조적 유사 지수(Structural Similarity Index, SSIM)을 사용했다. SSIM은 입력 데이터(x)와 정답 데이터(y) 간의 휘도, 대비, 구조에 대해 널리 사용되는 측정 값이며 다음과 같이 정의된다 (식 (2)).

-- 식 --  

이전의 선행연구가 없으므로, 비교할 기준모델 (baseline model)로는 SciPy 패키지의 선명화 효과 필터(sharpen filter)를 사용하였다.

## 7.  실험결과
-- loss 그림 --  
-- prediction 그림 --  








## 8. 결론 및 향후 연구 방향



