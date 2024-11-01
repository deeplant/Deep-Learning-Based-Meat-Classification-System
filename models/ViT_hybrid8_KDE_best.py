## MLflow 설정
port = 6002 # 포트 번호
experiment_name = "channel_KDE" # 실험 이름
run_name = "Marbling batch,vit KDE_correct_include4,nanX" # run name
save_model = True # 모델 저장

## 중요 파라미터
kfold = False # True: 5 fold-cross-val (5번), False: 4:1로 데이터를 분할해서 한 번만 학습
num_epochs = 20
batch_size = 16 # 수정
lr = 0.00001 # learning rate
weight_decay = 0.00005 # L2 Regularization
model_name = 'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k' # timm 모델 vit_base_patch32_clip_448.laion2b_ft_in12k_in1k
# vit_base_r50_s16_224.orig_in21k
image_resize = 512 # 이미지를 resize 한 후 input_size로 crop
input_size = 448 # 모델의 input_size
rotation_degree = 45 # transforms에서 랜덤 -n~n 만큼 이미지를 회전함 (데이터 증강)

## 그 외
mean=[0.4834, 0.3656, 0.3474] # 데이터셋의 mean과 std
std=[0.2097, 0.2518, 0.2559]
minibatch_count = 10 # 1 epoch마다 n번 train_loss를 기록
label_names = ['Marbling', 'Color', 'Texture', 'Surface Moisture', 'Total']

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import timm
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import vflip
import ast

# 제외할 이미지 목록 읽기 함수
def read_exclude_list(file_path):
    exclude_images = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    # 문자열을 리스트로 변환
                    image_list = ast.literal_eval(line.strip())
                    # 리스트의 각 이미지를 set에 추가
                    exclude_images.update(image_list)
                except:
                    # 라인을 파싱할 수 없는 경우 무시
                    continue
    return exclude_images

def add_flipped_images_to_dataset(df, grade='등심3'):
    flipped_rows = []
    
    for _, row in df[df['Grade'] == grade].iterrows():
        flipped_row = row.copy()
        flipped_row['is_flipped'] = True
        flipped_rows.append(flipped_row)
    
    df_flipped = pd.DataFrame(flipped_rows)
    df = pd.concat([df, df_flipped], ignore_index=True)
    
    print(f"Added flipped images for {grade}. Original count: {len(df) - len(flipped_rows)}, New total: {len(df)}")
    
    return df

def add_rotated_images(df, grade='등심3'):
    rotated_rows = []
    
    for _, row in df[df['Grade'] == grade].iterrows():
        rotated_row = row.copy()
        rotated_row['is_rotated'] = True
        rotated_rows.append(rotated_row)
    
    df_rotated = pd.DataFrame(rotated_rows)
    df = pd.concat([df, df_rotated], ignore_index=True)
    
    print(f"Added rotated images for {grade}. Original count: {len(df) - len(rotated_rows)}, New total: {len(df)}")
    
    return df

# '차이_4이상.txt'와 'nan_값_포함.txt' 파일에서 제외할 이미지 목록 읽기
exclude_file_path1 = './meat_dataset/차이_4이상.txt'
exclude_file_path2 = './meat_dataset/nan_값_포함.txt'
exclude_images = read_exclude_list(exclude_file_path1).union(read_exclude_list(exclude_file_path2))
exclude_images_4 = read_exclude_list(exclude_file_path1)
exclude_images_nan = read_exclude_list(exclude_file_path2)

def find_header_row(file_path):
    required_columns = ['No', '등급', 'Marbling(마블링정도)', 'Color(색깔)', 'Texture(조직감)', 'Surface Moisture(표면육즙)', 'Total(기호도)']
    
    # 엑셀 파일을 열고 각 행을 확인
    for i in range(20):  # 처음 20행만 확인 (필요에 따라 조정 가능)
        df = pd.read_excel(file_path, header=i, nrows=1)
        if all(col in df.columns for col in required_columns):
            return i
    
    raise ValueError(f"Required columns not found in the first 20 rows of {file_path}")

# 데이터 처리 함수
def process_data(base_directory, label_directory):
    
    label_directory = os.path.join(base_directory, 'labels')
    
    # 엑셀 파일 경로 설정
    excel_files = {
        '등심1++': os.path.join(label_directory, 'label_1++.xlsx'),
        '등심1+': os.path.join(label_directory, 'label_1+.xlsx'),
        '등심1': os.path.join(label_directory, 'label_1.xlsx'),
        '등심2': os.path.join(label_directory, 'label_2.xlsx'),
        '등심3': os.path.join(label_directory, 'label_3.xlsx')
    }

    # 이미지 파일 경로 설정
    image_directories = {
        '등심1++': os.path.join(base_directory, '등심1++'),
        '등심1+': os.path.join(base_directory, '등심1+'),
        '등심1': os.path.join(base_directory, '등심1'),
        '등심2': os.path.join(base_directory, '등심2'),
        '등심3': os.path.join(base_directory, '등심3')
    }
    
    dataframes = []
    for grade, file_path in excel_files.items():
        header_row = find_header_row(file_path)
        df = pd.read_excel(file_path, header=header_row)

        # 'No' 열이 비어 있지 않고, 다른 중요한 열들이 비어 있지 않은 행들만 유지
        df = df.dropna(subset=['No', '등급'])
        df = df.dropna(subset=['Marbling(마블링정도)', 'Color(색깔)', 'Texture(조직감)', 'Surface Moisture(표면육즙)', 'Total(기호도)'])

        # 숫자가 아닌 값 제거 ('측정불가' 등)
        columns_to_check = ['Marbling(마블링정도)', 'Color(색깔)', 'Texture(조직감)', 'Surface Moisture(표면육즙)', 'Total(기호도)']
        for column in columns_to_check:
            df = df[pd.to_numeric(df[column], errors='coerce').notnull()]

        # 모든 라벨 값에 2배 적용
        df['Marbling(마블링정도)'] = df['Marbling(마블링정도)'].astype(float) * 2
        df['Color(색깔)'] = df['Color(색깔)'].astype(float) * 2
        df['Texture(조직감)'] = df['Texture(조직감)'].astype(float) * 2
        df['Surface Moisture(표면육즙)'] = df['Surface Moisture(표면육즙)'].astype(float) * 2
        df['Total(기호도)'] = df['Total(기호도)'].astype(float) * 2

        # 등심1에 대해서만 No 값 조정
        if grade == '등심1':
            df['No'] = df['No'].apply(lambda x: x - 1 if x > 103 else x)

        # 이미지 경로 생성
        df['image_path'] = df['No'].apply(lambda x: os.path.join(image_directories[grade], f"{grade}_{int(x):05d}.jpg"))

        # 엑셀 파일의 번호와 이미지 파일의 번호가 일치하는 경우만 필터링
        df = df[df['image_path'].apply(os.path.exists)]

        # '차이_4이상.txt'와 'nan_값_포함.txt'에 있는 이미지 제외
        df = df[~df['image_path'].apply(lambda x: os.path.basename(x) in exclude_images_4)]

        # 등급 열 추가
        df['Grade'] = grade
        
        df['is_flipped'] = False
        df['is_rotated'] = False
        
        # 유효한 파일 경로가 몇 개인지 로그 출력
        max_no = df['No'].max()
        print(f"Filtered valid image paths for {grade}: {len(df)} / {max_no}")
        
        #if grade == '등심3':
        #    df = add_rotated_images(df)
        
        #if grade == '등심2':
        #    df = add_flipped_images_to_dataset(df, grade='등심2')
        if grade == '등심3':
            df = add_flipped_images_to_dataset(df)
        
        dataframes.append(df)

      # 모든 데이터프레임 병합
    all_data = pd.concat(dataframes, ignore_index=True)
    all_data.columns = all_data.columns.str.replace(r'\(.*\)', '', regex=True).str.strip()
    all_data.columns = all_data.columns.str.replace(' ', '_')
    all_data = all_data[['image_path', 'No', 'Grade', 'Marbling', 'Color', 'Texture', 'Surface_Moisture', 'Total', 'is_flipped', 'is_rotated']]
    
    return all_data

# 경로 설정
base_directory = './meat_dataset'
label_directory = os.path.join(base_directory, 'labels')

all_data = process_data(base_directory, label_directory)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

n_splits = 5
fold_data = [[] for _ in range(n_splits)]
label_columns = ['Marbling', 'Color', 'Texture', 'Surface_Moisture', 'Total']

for grade in all_data['Grade'].unique():
    grade_data = all_data[all_data['Grade'] == grade].copy()
    
    # 라벨 정규화
    scaler = StandardScaler()
    normalized_labels = scaler.fit_transform(grade_data[label_columns])
    
    # 정규화된 라벨의 평균을 계산하여 새로운 점수 생성
    grade_data['combined_score'] = np.mean(normalized_labels, axis=1)
    
    # 결합된 점수를 기준으로 정렬
    grade_data = grade_data.sort_values('combined_score')
    
    # 5개의 동일한 크기의 그룹으로 나눔
    grade_data['group'] = pd.qcut(grade_data['combined_score'], q=n_splits, labels=False)
    
    # 각 그룹을 5개의 폴드에 순차적으로 할당
    for i, row in enumerate(grade_data.itertuples()):
        fold_index = i % n_splits
        fold_data[fold_index].append(row)
        
    # 각 폴드의 데이터를 DataFrame으로 변환
fold_data = [pd.DataFrame(fold) for fold in fold_data]

    # 인덱스 재설정 및 불필요한 열 제거
for fold in fold_data:
    fold.reset_index(drop=True, inplace=True)
    fold.drop(columns=['Index', 'group', 'combined_score'], inplace=True)

print(f"total dataset: {sum(len(fold) for fold in fold_data[:5])}")
print(f"fold 0: {len(fold_data[0])}")
print(f"fold 1: {len(fold_data[1])}")
print(f"fold 2: {len(fold_data[2])}")
print(f"fold 3: {len(fold_data[3])}")
print(f"fold 4: {len(fold_data[4])}")    

    # 교차 검증을 위한 데이터 분할 예시 (0번째 폴드를 검증 세트로 사용)
val_data = fold_data[0]
train_data = pd.concat([fold_data[i] for i in range(1, n_splits)], ignore_index=True)



print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")

# 각 등급별 데이터 분포 확인
for grade in all_data['Grade'].unique():
    print(f"\nGrade: {grade}")
    print(f"Train: {len(train_data[train_data['Grade'] == grade])}")
    print(f"Validation: {len(val_data[val_data['Grade'] == grade])}")

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from scipy.stats import gaussian_kde

# RGB 이미지를 위한 train transform
rgb_train_transform = transforms.Compose([
    transforms.RandomRotation(rotation_degree),
    transforms.Resize(image_resize),
    transforms.CenterCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# RGB 이미지를 위한 validation transform
rgb_val_transform = transforms.Compose([
    transforms.Resize(image_resize),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# KDE 이미지를 위한 transform (train과 validation 동일)
kde_transform = transforms.Compose([
    transforms.Resize(input_size),
#     transforms.CenterCrop(input_size),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
])

class MeatDataset(Dataset):
    def __init__(self, dataframe, rgb_transform, kde_transform, kde_dir='./meat_dataset/ku_KDE/graph_picture',is_train = False):
#         self.data = dataframe
        self.rgb_transform = rgb_transform
        self.kde_transform = kde_transform
        self.kde_dir = kde_dir
        self.is_train = is_train
        
        if not self.is_train:
            self.data = dataframe[~(dataframe['is_flipped'] | dataframe['is_rotated'])]
        else:
            self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.data.iloc[idx]
        
        
        
        # RGB 이미지 로드 및 transform 적용
        img_path = self.data.iloc[idx]['image_path']
        rgb_image = Image.open(img_path).convert('RGB')
        
        if row.get('is_flipped', False) and self.is_train:
            rgb_image = vflip(rgb_image)

        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
            
        if row.get('is_rotated', False) and self.is_train:
            rgb_image = rotate(rgb_image, 90)

        # KDE 이미지 로드 및 transform 적용
        grade = self.data.iloc[idx]['Grade'].replace('등심', '')
        kde_path = os.path.join(self.kde_dir, f'kde_Marbling_{grade}_grade.png')
        kde_image = Image.open(kde_path).convert('L')
        if self.kde_transform:
            kde_image = self.kde_transform(kde_image)

        # 이미지 결합
        combined_image = torch.cat([rgb_image, kde_image], dim=0)

        # 라벨 추출
        # labels = self.data.iloc[idx][['Marbling', 'Color', 'Texture', 'Surface_Moisture', 'Total']].values.astype(np.float32)
        label_columns = ['Marbling', 'Color', 'Texture', 'Surface_Moisture', 'Total']
        labels = [row.get(col, float('nan')) for col in label_columns]
        labels = torch.tensor([row.get(col, float('nan')) for col in label_columns], dtype=torch.float)
        return combined_image, labels.clone().detach()

# 데이터셋 초기화
train_dataset = MeatDataset(dataframe=train_data, rgb_transform=rgb_train_transform, kde_transform=kde_transform,is_train=True)
val_dataset = MeatDataset(dataframe=val_data, rgb_transform=rgb_val_transform, kde_transform=kde_transform,is_train=False)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)



# train과 val 데이터셋의 개수 출력
print(f"Number of samples in train dataset: {len(train_dataset)}")
print(f"Number of samples in validation dataset: {len(val_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import torch
import torch.nn as nn
import timm

class CustomViTRegressor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.num_features = self.base_model.num_features

        # 마블링, 조직감, 기호도를 위한 MLP
        self.marbling_texture_preference = nn.Sequential(
            nn.Linear(self.num_features, 256),
#             nn.Dropout(p = 0.25),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
        # 표면육즙을 위한 MLP
        self.moisture_head = nn.Sequential(
            nn.Linear(self.num_features, 64),
#             nn.Dropout(p = 0.25),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 색깔을 위한 MLP
        self.color_head = nn.Sequential(
            nn.Linear(self.num_features, 64),
#             nn.Dropout(p = 0.25),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        base_output = self.base_model(x)
        
        features = base_output

        marbling_texture_preference = self.marbling_texture_preference(features)
        moisture = self.moisture_head(features)
        color = self.color_head(features)
        
        # 원래의 라벨 순서대로 출력을 결합
        return torch.cat([
            marbling_texture_preference[:, 0:1],  # 마블링
            color,  # 색깔
            marbling_texture_preference[:, 1:2],  # 조직감
            moisture,  # 표면육즙
            marbling_texture_preference[:, 2:3]  # 기호도
        ], dim=1)

def create_model(model_name, device):
    base_model = timm.create_model(model_name, pretrained=True, num_classes=0, in_chans = 4)
    model = CustomViTRegressor(base_model)
    model = model.to(device)
    
    return model

# Usage example:
model = create_model(model_name, device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# 학습 및 검증 손실 리스트 초기화
training_loss = []
validation_loss = []
accuracy_list = []
best_accuracy = 0.0
best_epoch = 0 

# 학습 루프
minibatch = len(train_loader)//minibatch_count # 1 epoch 마다 10번의 minibatch마다 기록
train_losses = []
val_losses = []
train_r2 = []
val_r2 = []
train_acc = []
val_acc = []
average_train_acc = []
average_val_acc = []

# MLflow 설정
mlflow.set_tracking_uri("http://0.0.0.0:"+str(port))  # MLflow 서버 URI
mlflow.set_experiment(experiment_name)  # 실험 이름 설정

# 모델 학습
with mlflow.start_run(run_name=run_name):
    # MLflow에 파라미터 저장
    mlflow.log_param("model", model_name)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("weight_decay", weight_decay)
    mlflow.log_param("rotation_degree", rotation_degree)
    mlflow.log_param("kfold", kfold)

    if kfold:
        n_folds = 5
        print("\n" + "="*50)
        print("     K-Fold Cross Validation (K=5) Enabled")
        print("="*50 + "\n")
    else:
        n_folds = 1
        print("\n" + "="*50)
        print("     Single Fold Training")
        print("="*50 + "\n")

    all_train_losses = []
    all_val_losses = []
    all_train_r2 = []
    all_val_r2 = []
    all_train_acc = []
    all_val_acc = []

    for fold in range(n_folds):
        if kfold:
            print(f"\n{'='*20} Fold {fold+1}/{n_folds} {'='*20}\n")
            
            # 5-fold cross validation 데이터 설정
            train_data = pd.concat([fold_data[i] for i in range(5) if i != fold])
            val_data = fold_data[fold]
            train_dataset = MeatDataset(train_data, transform=train_transform)
            val_dataset = MeatDataset(val_data, transform=val_transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            # 모델 초기화
            model = create_model()
            
            # 손실 함수 및 옵티마이저 정의
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_r2 = []
        val_r2 = []
        train_acc = []
        val_acc = []
        
        for epoch in range(num_epochs): # epoch만큼 학습
            print(f"\n{'-'*15} Epoch {epoch+1}/{num_epochs} {'-'*15}", end='')
            if kfold:
                print(f" (Fold {fold+1}/{n_folds})\n")
            else:
                print()
            
            model.train()
            running_loss = 0.0
            all_outputs = []
            all_labels = []
            i = 0
            for images, labels in tqdm(train_loader): # 학습 루프
                images, labels = images.to(device), labels.to(device).float()
                i += 1

                # step (backprop)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels) # MSE로 정답 값과 예측 값 사이의 Loss 계산
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                all_outputs.append(outputs.cpu().detach().numpy()) # 모든 output을 저장 (5개의 값)
                all_labels.append(labels.cpu().detach().numpy()) # 모든 label을 저장 (5개의 값)

                if i % minibatch == 0:  # Dataset 크기의 1/10의 minibatch마다 loss 값을 기록
                    train_loss = running_loss / i # loss 계산
                    step = epoch * len(train_loader) + i + fold * num_epochs * len(train_loader)
                    #print(f'[{epoch + 1}, {i}] train_loss: {train_loss:.3f}')
                    mlflow.log_metric('train_loss', train_loss, step=step) # MLflow에 기록

            all_outputs = np.concatenate(all_outputs, axis=0) # batch당 모든 output을 더해줌
            all_labels = np.concatenate(all_labels, axis=0) # batch당 모든 label을 더해줌
            train_r2_value = r2_score(all_labels, all_outputs, multioutput='uniform_average')  # 통합 R2 score

            # 정확도 계산
            def calculate_accuracy(labels, outputs):
                accuracies = []
                for i in range(labels.shape[1]):
                    correct = ((labels[:, i] - 1) <= outputs[:, i]) & (outputs[:, i] <= (labels[:, i] + 1))
                    accuracy = correct.sum() / len(labels)
                    accuracies.append(accuracy)
                return accuracies

            train_accuracies = calculate_accuracy(all_labels, all_outputs)
            average_train_accuracies = np.mean(train_accuracies)

            train_losses.append(running_loss / len(train_loader)) # epoch마다 train loss 저장
            train_acc.append(train_accuracies) # epoch마다 각 라벨의 정확도 저장
            average_train_acc.append(average_train_accuracies) # epoch마다 평균 정확도 저장

            print(f"\nEpoch {epoch+1} Summary:")
            if kfold:
                print(f"Fold: {fold+1}/{n_folds}")
            print(f"Train Loss: {running_loss / len(train_loader):.3f}")
            print(f"Train R2: {train_r2_value:.3f}")
            print(f"Train Accuracies: {train_accuracies}")
            print(f"Average Train Accuracy: {average_train_accuracies:.3f}")
            
            train_r2.append(train_r2_value)

            # 검증 루프
            model.eval()
            val_loss = 0.0
            all_val_outputs = []
            all_val_labels = []
            with torch.no_grad(): # validation은 학습을 안하기 때문에 기울기 계산 안함
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).float()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    all_val_outputs.append(outputs.cpu().detach().numpy())
                    all_val_labels.append(labels.cpu().detach().numpy())

            val_loss /= len(val_loader)
            all_val_outputs = np.concatenate(all_val_outputs, axis=0)
            all_val_labels = np.concatenate(all_val_labels, axis=0)
            val_r2_value = r2_score(all_val_labels, all_val_outputs, multioutput='uniform_average')  # 통합 R2 score

            # 정확도 계산
            val_accuracies = calculate_accuracy(all_val_labels, all_val_outputs)
            average_val_accuracies = np.mean(val_accuracies)

            print(f"\nValidation Results:")
            if kfold:
                print(f"Fold: {fold+1}/{n_folds}")
            print(f"Epoch: {epoch+1}/{num_epochs}")
            print(f"Validation Loss: {val_loss:.3f}")
            print(f"Validation R2: {val_r2_value:.3f}")
            print(f"Validation Accuracies: {val_accuracies}")
            print(f"Average Validation Accuracy: {average_val_accuracies:.3f}")
            
            mlflow.log_metric('val_loss', val_loss, step=epoch + 1 + fold * num_epochs)
            mlflow.log_metric('val_R2', val_r2_value, step=epoch + 1 + fold * num_epochs)

            for i, acc in enumerate(val_accuracies): # 각 라벨마다 정확도 출력 
                label_name = label_names[i]
                mlflow.log_metric(f'val_acc_{label_name}', acc, step=epoch + 1 + fold * num_epochs)

            mlflow.log_metric('average_val_acc', average_val_accuracies, step=epoch + 1 + fold * num_epochs)

            # 가장 낮은 val_loss를 가진 모델 저장
            if val_loss < best_val_loss and save_model:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(model, "best_model")

            val_losses.append(val_loss)
            val_r2.append(val_r2_value)
            val_acc.append(val_accuracies)
            average_val_acc.append(average_val_accuracies)
            
            
        # 각 fold 또는 단일 실행의 결과 저장
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_r2.append(train_r2)
        all_val_r2.append(val_r2)
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)

    if kfold:
        # k-fold의 평균 계산 및 기록
        avg_train_losses = np.mean(all_train_losses, axis=0)
        avg_val_losses = np.mean(all_val_losses, axis=0)
        avg_train_r2 = np.mean(all_train_r2, axis=0)
        avg_val_r2 = np.mean(all_val_r2, axis=0)
        avg_train_acc = np.mean(all_train_acc, axis=0)
        avg_val_acc = np.mean(all_val_acc, axis=0)

        for epoch in range(num_epochs):
            mlflow.log_metric('avg_train_loss', avg_train_losses[epoch], step=epoch)
            mlflow.log_metric('avg_val_loss', avg_val_losses[epoch], step=epoch)
            mlflow.log_metric('avg_train_R2', avg_train_r2[epoch], step=epoch)
            mlflow.log_metric('avg_val_R2', avg_val_r2[epoch], step=epoch)
            
            for i, label_name in enumerate(label_names):
                mlflow.log_metric(f'avg_train_acc_{label_name}', avg_train_acc[epoch][i], step=epoch)
                mlflow.log_metric(f'avg_val_acc_{label_name}', avg_val_acc[epoch][i], step=epoch)

            # k-fold 정확도의 평균 (각 라벨의 정확도를 먼저 평균 낸 후, 이를 다시 fold 간 평균)
            mlflow.log_metric('kfold_avg_train_acc', np.mean(avg_train_acc[epoch]), step=epoch)
            mlflow.log_metric('kfold_avg_val_acc', np.mean(avg_val_acc[epoch]), step=epoch)
        
    if kfold:
        # 가장 높은 validation R2 score 출력
        best_r2 = -float('inf')
        best_fold = -1
        best_epoch = -1
        best_accuracies = None
        best_loss = None

        for fold in range(n_folds):
            for epoch, r2 in enumerate(all_val_r2[fold]):
                if r2 > best_r2:
                    best_r2 = r2
                    best_fold = fold
                    best_epoch = epoch
                    best_accuracies = all_val_acc[fold][epoch]
                    best_loss = all_val_losses[fold][epoch]

        print("\nBest Validation Performance:")
        print(f"Fold: {best_fold + 1}")
        print(f"Epoch: {best_epoch + 1}")
        print("Accuracies for each label:")
        for i, label_name in enumerate(label_names):
            print(f"  {label_name}: {best_accuracies[i]:.4f}")
        print(f"Average Accuracy: {np.mean(best_accuracies):.4f}")
        print(f"Loss: {best_loss:.4f}")
        print(f"R2 Score: {best_r2:.4f}")

        # MLflow에 최고 성능 기록
        mlflow.log_metric("best_val_r2", best_r2)
        mlflow.log_metric("best_val_loss", best_loss)
        mlflow.log_metric("best_val_avg_accuracy", np.mean(best_accuracies))
        for i, label_name in enumerate(label_names):
            mlflow.log_metric(f"best_val_acc_{label_name}", best_accuracies[i])

    else:
        best_r2 = max(val_r2)
        best_epoch = val_r2.index(best_r2)
        best_accuracies = val_acc[best_epoch]
        best_loss = val_losses[best_epoch]

        print("\nBest Validation Performance:")
        print(f"Epoch: {best_epoch + 1}")
        print("Accuracies for each label:")
        for i, label_name in enumerate(label_names):
            print(f"  {label_name}: {best_accuracies[i]:.4f}")
        print(f"Average Accuracy: {np.mean(best_accuracies):.4f}")
        print(f"Loss: {best_loss:.4f}")
        print(f"R2 Score: {best_r2:.4f}")

        # MLflow에 최고 성능 기록
        mlflow.log_metric("best_val_r2", best_r2)
        mlflow.log_metric("best_val_loss", best_loss)
        mlflow.log_metric("best_val_avg_accuracy", np.mean(best_accuracies))
        for i, label_name in enumerate(label_names):
            mlflow.log_metric(f"best_val_acc_{label_name}", best_accuracies[i])






