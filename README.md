Deeplant
--------
딥러닝기반 육류 맛 선호 예측 시스템 개발 프로젝트.
육류 단면 이미지 기반의 등급 판정, 관능평가 딥러닝 모델 개발 및 웹 관리 시스템 개발

시스템 소개
---
<img width="735" alt="스크린샷 2024-08-21 오후 5 00 54" src="https://github.com/user-attachments/assets/a44f9a56-bb74-4048-9d81-9e14c77bf6c3">

관능평가 데이터
---------
<img width="231" alt="스크린샷 2024-08-21 오후 5 02 04" src="https://github.com/user-attachments/assets/b9d6c397-0f6f-48d4-817f-843c5509abfd">

마블링, 육색, 조직감, 표면육즙, 기호도 5가지 값에 대한 Regression Task


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

'models/' 폴더에 custom model code를 적을 .py 파일 생성.
pytorch 기반의 모델 클래스 제작.
제작한 class를 return 하는 외부에서 접근 가능한 create_model() 함수 추가.

Configuration file 작성
---
1. 특정 configuration file 복사.
2. 시스템 설명서의 configuration file 설명을 보면서 상황에 맞게 작성.

Create Custom Loss
---
'utils/step.py'에 custom loss 코드 작성 권장.

결과 및 성능
---
![image](https://github.com/user-attachments/assets/60b3698e-8dba-42dc-a85a-91b34b322841)
2330개의 데이터 학습 기준

acc 05: 모델의 예측값이 정답의 +-0.5 이내일 경우 정답으로 처리한 정확도

acc 10: 모델의 예측값이 정답의 +-1.0 이내일 경우 정답으로 처리한 정확도




