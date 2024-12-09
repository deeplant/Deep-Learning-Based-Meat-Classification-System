#Deeplant
---
딥러닝기반 육류 맛 선호 예측 시스템 개발 프로젝트.
육류 단면 이미지 기반의 등급 판정, 관능평가 딥러닝 모델 개발 및 웹 관리 시스템 개발

##프로젝트 목표

육류에 대한 고객/구매자의 '맛에 대한 선호도(tasty preference)' 데이터를 육류 이미지와 분석된 미각 데이타로 나누어 체계적으로 수집하고, 이를 기반으로 이미지 기반의 육류에 관한 상세한 맛 데이타를 자동으로 분류하고 예측하는 인공지능 예측 모델을 구축하여 향후 고객/구매자들에게 최적화된 육류 개인화 추천을 위한 시스템 개발을 목표로 합니다.

시스템 소개
---
<img width="735" alt="스크린샷 2024-08-21 오후 5 00 54" src="https://github.com/user-attachments/assets/a44f9a56-bb74-4048-9d81-9e14c77bf6c3">

관능평가 데이터
---------
<img width="231" alt="스크린샷 2024-08-21 오후 5 02 04" src="https://github.com/user-attachments/assets/b9d6c397-0f6f-48d4-817f-843c5509abfd">


Installation
-----
Ensure you have a python environment, python=4.5.4.60 is recommended. cuda를 사용가능한 Nvdia GPU 추천.

'''
pip install~
'''

Quick Start
----
1. clone repository
'''
git clone ~
'''
2. Run Model
'''
python
'''

Additional
---
Create Custom Model
---
기본적인 pytorch 모델 제작법과 같다. pytorch 모델 제작법은 공식 document 참고.

'ml_training/models' 폴더에 custom model code를 적을 .py 파일 생성.
pytorch 기반의 모델 클래스 제작.
모델 클래스의 forward 부분에 들어오는 입력 값이 list이므로 이 부분을 주의해서 코드 작성.
클래스 안에 getAlgorithm 함수 추가. classification or regression 리턴.
제작한 class를 return 하는 외부에서 접근 가능한 create_model() 함수 추가.

Configuration file 작성
---
1. 특정 configuration file 복사.
2. 시스템 설명서의 configuration file 설명을 보면서 상황에 맞게 작성.

Create Custom Loss
---
'ml_training/loss/loss.py'에 custom loss 코드 작성 권장.







