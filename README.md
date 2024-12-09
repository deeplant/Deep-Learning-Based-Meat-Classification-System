# Deeplant
딥러닝기반 육류 맛 선호 예측 시스템 개발 프로젝트.
육류 단면 이미지 기반의 등급 판정, 관능평가 딥러닝 모델 개발 및 웹 관리 시스템 개발

## 프로젝트 목표

육류에 대한 고객/구매자의 '맛에 대한 선호도(tasty preference)' 데이터를 육류 이미지와 분석된 미각 데이타로 나누어 체계적으로 수집하고, 이를 기반으로 이미지 기반의 육류에 관한 상세한 맛 데이타를 자동으로 분류하고 예측하는 인공지능 예측 모델을 구축하여 향후 고객/구매자들에게 최적화된 육류 개인화 추천을 위한 시스템 개발을 목표로 합니다.

## 시스템 소개

<img width="735" alt="스크린샷 2024-08-21 오후 5 00 54" src="https://github.com/user-attachments/assets/a44f9a56-bb74-4048-9d81-9e14c77bf6c3">

## 관능평가 데이터

<img width="231" alt="스크린샷 2024-08-21 오후 5 02 04" src="https://github.com/user-attachments/assets/b9d6c397-0f6f-48d4-817f-843c5509abfd">


# Installation

Ensure you have a python environment, python=4.5.4.60 is recommended. cuda를 사용가능한 Nvdia GPU 추천.

'''
pip install~
'''

# Quick Start ⚡
1. clone repository
'''
git clone https://github.com/skitw427/20242R0136COSE48002.git
'''
2. Run Model (original)
'''
python train.py --experiment "실험 이름" --run "run 이름"
'''

# Additional

## Create Custom Model

기본적인 pytorch 모델 제작법과 같다. pytorch 모델 제작법은 공식 document 참고.

1. 'ml_training/models' 폴더에 custom model code를 적을 .py 파일 생성.
2. pytorch 기반의 모델 클래스 제작.
3. 모델 클래스의 forward 부분에 들어오는 입력 값이 list이므로 이 부분을 주의해서 코드 작성.
4. 클래스 안에 getAlgorithm 함수 추가. classification or regression 리턴.
5. 제작한 class를 return 하는 외부에서 접근 가능한 create_model() 함수 추가.

## Configuration file 작성
1. 특정 configuration file 복사.
2. 시스템 설명서의 configuration file 설명을 보면서 상황에 맞게 작성.

## Create Custom Loss
'ml_training/loss/loss.py'에 custom loss 코드 작성 권장.

# 사용 모델
- ViT
- CNN
- CoAtNet
- Swin Transformer

# 사용 데이터셋
## 등급예측
| Name | Data Type | Range |
| --- | --- | --- |
| image src | string |
| grade | string | 1++,1+,1,2,3 |
- AI hub에서 제공하는 75000개의 육류 이미지 사용

## 맛데이터 예측
|Name|Data Type|Range|
| --- | --- | --- |
|image src|string|
|grade|string|1++,1+,1,2,3|
|color|float|1 ~ 10|
|marbling|float|1 ~ 10|
|texture|float|1 ~ 10|
|surface moisture|float|1 ~ 10|
|total|float|1 ~ 10|
|grade|float|1 ~ 10|

- 5161개의 육류 이미지 사용

# 결과 및 성능
## 육류 등급 예측
| Models | # of Params | Accuracy |
| --- | --- | --- |
| vit_base_r50_s16_224.orig_in21k | 97.9M | 98.8 |

## 맛 등급 예측
|Models|# of Params|R2 Score|Average ACC|
| --- | --- | --- | --- |
|vit_base_r50_s16_224.orig_in21k|~|~|

# manage.py start Argument
|args|용도|
| --- | --- |
|run|mlflow 이름 설정|
|experiment|mlflow experiment 이름 설정|
|model_cfgs|모델 configuration file 경로 설정|
|mode|train or test 모드 설정|
|epochs|학습 시 반복 횟수 설정|
|log_epoch | 몇번 반복 시 모델을 저장할 것인지 설정|
|lr | learning rate를 의미하며 학습 결과를 모델에 얼마나 반영할 것인지를 설정|
|data_path |csv가 저장된 폴더의 경로|
|csv_name|csv 파일 이름|
|sanity|boolean 값을 의미하며 True면 코드의 작동만 확인하기 위해 한 배치만 학습하고 나머지는 스킵함|











