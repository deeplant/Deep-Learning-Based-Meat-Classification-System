## MLflow 설정
port = 6001 # 포트 번호
experiment_name = "New ViT_hybrid" # 실험 이름
run_name = "3-aug + attmap-m-c secimg" # run name

## 중요 파라미터
kfold = False # True: 5 fold-cross-val (5번), False: 4:1로 데이터를 분할해서 한 번만 학습
num_epochs = 30
batch_size = 16
lr = 0.00001 # learning rate
weight_decay = 0.00005 # L2 Regularization
model_name = 'vit_base_r50_s16_224.orig_in21k' # timm 모델
image_resize = 256 # 이미지를 resize 한 후 input_size로 crop
input_size = 224 # 모델의 input_size
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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.transforms.functional import vflip, rotate

import os
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 경로 설정
base_directory = './meat_dataset'
label_directory = os.path.join(base_directory, 'labels')

# 엑셀 파일 경로 설정
excel_files = {
    '등심1++': os.path.join(label_directory, 'label_1++.xlsx'),
    '등심1+': os.path.join(label_directory, 'label_1+.xlsx'),
    '등심1': os.path.join(label_directory, 'label_1.xlsx'),
    '등심2': os.path.join(label_directory, 'label_2.xlsx'),
    '등심3': os.path.join(label_directory, 'label_3.xlsx')
}

def find_header_row(file_path):
    required_columns = ['No', '등급', 'Marbling(마블링정도)', 'Color(색깔)', 'Texture(조직감)', 'Surface Moisture(표면육즙)', 'Total(기호도)']
    
    # 엑셀 파일을 열고 각 행을 확인
    for i in range(20):  # 처음 20행만 확인 (필요에 따라 조정 가능)
        df = pd.read_excel(file_path, header=i, nrows=1)
        if all(col in df.columns for col in required_columns):
            return i
    
    raise ValueError(f"Required columns not found in the first 20 rows of {file_path}")

# 이미지 파일 경로 설정 (원본 이미지와 단면 이미지)
image_directories = {grade: os.path.join(base_directory, grade) for grade in ['등심1++', '등심1+', '등심1', '등심2', '등심3']}
section_directories = {grade: os.path.join(base_directory, f'sectionV4_{grade}') for grade in ['등심1++', '등심1+', '등심1', '등심2', '등심3']}

# 데이터 로딩 및 전처리
dataframes = []
for grade, file_path in excel_files.items():
    header_row = find_header_row(file_path)
    df = pd.read_excel(file_path, header=header_row)
    
    df = df.dropna(subset=['No'])
    df = df.dropna(subset=['Marbling(마블링정도)', 'Color(색깔)', 'Texture(조직감)', 'Surface Moisture(표면육즙)', 'Total(기호도)'])
    
    columns_to_check = ['Marbling(마블링정도)', 'Color(색깔)', 'Texture(조직감)', 'Surface Moisture(표면육즙)', 'Total(기호도)']
    for column in columns_to_check:
        df = df[pd.to_numeric(df[column], errors='coerce').notnull()]
    
    for column in columns_to_check:
        df[column] = df[column].astype(float) * 2
    
    if grade == '등심1':
        df['No'] = df['No'].apply(lambda x: x - 1 if x > 103 else x)
    
    df['image_path'] = df['No'].apply(lambda x: os.path.join(image_directories[grade], f"{grade}_{int(x):05d}.jpg"))
    df['section_path'] = df['No'].apply(lambda x: os.path.join(section_directories[grade], f"{grade}_{int(x):05d}.jpg"))
    
    df = df[df['image_path'].apply(os.path.exists) & df['section_path'].apply(os.path.exists)]
    
    df = df[~df['image_path'].apply(lambda x: os.path.basename(x) in exclude_images_4)]
    
    df['Grade'] = grade
    
    df['is_flipped'] = False
    df['is_rotated'] = False
    
    print(f"Filtered valid image paths for {grade}: {len(df)} / {df['No'].max()}")
    
    if grade == '등심3':
        df = add_rotated_images(df)

    if grade == '등심2':
        df = add_flipped_images_to_dataset(df, grade='등심2')
    if grade == '등심3':
        df = add_flipped_images_to_dataset(df)

    dataframes.append(df)

all_data = pd.concat(dataframes, ignore_index=True)
all_data.columns = all_data.columns.str.replace(r'\(.*\)', '', regex=True).str.strip()
all_data.columns = all_data.columns.str.replace(' ', '_')
all_data = all_data[['image_path', 'section_path', 'No', 'Grade', 'Marbling', 'Color', 'Texture', 'Surface_Moisture', 'Total', 'is_flipped', 'is_rotated']]

    # 각 등급별로 train/val 분할
train_data = []
val_data = []

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

class MeatDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_train=False):
        self.dataframe = dataframe
        self.transform = transform
        self.is_train = is_train
        
        if not self.is_train:
            self.dataframe = dataframe[~(dataframe['is_flipped'] | dataframe['is_rotated'])]
        else:
            self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.dataframe.iloc[idx]['image_path']
        section_name = self.dataframe.iloc[idx]['section_path']
        image = Image.open(img_name).convert('RGB')
        
        row = self.dataframe.iloc[idx]
        
        if row.get('is_flipped', False) and self.is_train:
            image = vflip(image)
        
        if self.transform:
            image = self.transform(image)
        
        if row.get('is_rotated', False) and self.is_train:
            image = rotate(image, 90)
        
        label_columns = ['Marbling', 'Color', 'Texture', 'Surface_Moisture', 'Total']
        labels = [row.get(col, float('nan')) for col in label_columns]
        labels = torch.tensor(labels, dtype=torch.float)
        
        return image, labels, section_name

# 이미지 변환
train_transform = transforms.Compose([
    transforms.RandomRotation(rotation_degree),
    transforms.Resize(image_resize),
    transforms.CenterCrop(input_size),
    #transforms.RandomCrop(input_size),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=mean, std=std), # 데이터셋의 mean과 std (데이터셋이 바뀌면 계산해줘야 됨)
])
val_transform = transforms.Compose([
    transforms.Resize(image_resize),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_dataset = MeatDataset(dataframe=train_data, transform=train_transform, is_train=True)
val_dataset = MeatDataset(dataframe=val_data, transform=val_transform, is_train=False)

def collate_fn(batch):
    images, labels, section_paths = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels, section_paths

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

# train과 val 데이터셋의 개수 출력
print(f"Number of samples in train dataset: {len(train_dataset)}")
print(f"Number of samples in validation dataset: {len(val_dataset)}")

# for images, labels in train_loader: # 데이터 확인
#     print(f"Train - Images: {images.size()}, Labels: {labels.size()}")
#     break
# for images, labels in val_loader:
#     print(f"Validation - Images: {images.size()}, Labels: {labels.size()}")
#     break

import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class CustomViTRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = model_name
        self.num_features = self.base_model.num_features
        
        # row와 col 벡터를 처리하기 위한 선형 레이어 추가
        self.row_linear = nn.Linear(32 * 64, 128)  # 32x64 -> 128
        self.col_linear = nn.Linear(32 * 64, 128)  # 32x64 -> 128
        
        # 마블링을 위한 MLP (행과 열 벡터 추가)
        self.marbling_head = nn.Sequential(
            nn.Linear(self.num_features + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 조직감과 기호도를 위한 MLP
        self.texture_preference = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
        # 표면육즙을 위한 MLP
        self.moisture_head = nn.Sequential(
            nn.Linear(self.num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 색깔을 위한 MLP (HSV 벡터 포함)
        self.color_head = nn.Sequential(
            nn.Linear(self.num_features + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x, section_paths):
        features = self.base_model(x)
        
        # HSV 벡터 추출 (단면 이미지 사용)
        hsv_vector = self.get_hsv_vector(section_paths)
        
        # 행과 열 벡터 추출 (단면 이미지 사용)
        combined_vectors = self.extract_row_col_vectors(section_paths)

        # 마블링 예측 (행과 열 벡터 결합)
        marbling_features = torch.cat([features, combined_vectors], dim=1)
        marbling = self.marbling_head(marbling_features)

        # 조직감과 기호도 예측
        texture_preference = self.texture_preference(features)

        # 표면육즙 예측
        moisture = self.moisture_head(features)

        # 색깔 예측 (HSV 벡터 포함)
        color_features = torch.cat([features, hsv_vector], dim=1)
        color = self.color_head(color_features)
        
        # 원래의 라벨 순서대로 출력을 결합
        return torch.cat([
            marbling,
            color,
            texture_preference[:, 0:1],
            moisture,
            texture_preference[:, 1:2]
        ], dim=1)

    def get_hsv_vector(self, section_paths, epsilon=1e-6, target_min=-2.2, target_max=2.0, target_mean=-0.105):
        hsv_vectors = []

        for path in section_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # RGB에서 HSV로 변환
            hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # 히스토그램 계산
            hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 4, 8], [0, 180, 0, 256, 0, 256])
            hist_vector = hist.flatten()
            
            # 모든 빈(bin)에 작은 값을 추가하여 0을 방지
            hist_vector += epsilon
            
            # 정규화
            hist_vector = hist_vector / np.sum(hist_vector)
            
            # 벡터를 ViT 출력 범위로 조정
            current_min = hist_vector.min()
            current_max = hist_vector.max()
            hist_scaled = (hist_vector - current_min) / (current_max - current_min)
            hist_scaled = hist_scaled * (target_max - target_min) + target_min
            
            # 평균 조정
            mean_diff = target_mean - hist_scaled.mean()
            hist_adjusted = hist_scaled + mean_diff
            
            hsv_vectors.append(hist_adjusted)
        
        hsv_vectors = np.array(hsv_vectors)

        return torch.from_numpy(hsv_vectors).to(device)
    
    def extract_row_col_vectors(self, section_paths):
        row_vectors_list = []
        col_vectors_list = []
        
        for path in section_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            non_black_mask = img > 0

            rows = [img[j, non_black_mask[j, :]] for j in range(img.shape[0]) if np.any(non_black_mask[j, :])]
            cols = [img[non_black_mask[:, j], j] for j in range(img.shape[1]) if np.any(non_black_mask[:, j])]

            def normalize_and_resize(vector, length):
                if len(vector) == 0:
                    return np.zeros(length)
                vector = (vector - vector.min()) / (vector.max() - vector.min() + 1e-8)
                return np.interp(np.linspace(0, len(vector) - 1, length), np.arange(len(vector)), vector)

            # 64로 크기 변경
            row_vectors_batch = [normalize_and_resize(row, 64) for row in rows]
            col_vectors_batch = [normalize_and_resize(col, 64) for col in cols]

            # 32개로 맞추기
            if len(row_vectors_batch) > 32:
                row_indices = np.linspace(0, len(row_vectors_batch) - 1, 32, dtype=int)
                row_vectors_batch = [row_vectors_batch[j] for j in row_indices]
            elif len(row_vectors_batch) < 32:
                row_vectors_batch.extend([np.zeros(64)] * (32 - len(row_vectors_batch)))

            if len(col_vectors_batch) > 32:
                col_indices = np.linspace(0, len(col_vectors_batch) - 1, 32, dtype=int)
                col_vectors_batch = [col_vectors_batch[j] for j in col_indices]
            elif len(col_vectors_batch) < 32:
                col_vectors_batch.extend([np.zeros(64)] * (32 - len(col_vectors_batch)))

            row_vectors_list.append(np.array(row_vectors_batch))
            col_vectors_list.append(np.array(col_vectors_batch))

        # NumPy 배열로 변환 후 PyTorch 텐서로 변환
        row_vectors = torch.from_numpy(np.array(row_vectors_list)).to(device).float()
        col_vectors = torch.from_numpy(np.array(col_vectors_list)).to(device).float()

        # 형태 변경 및 선형 레이어 통과
        batch_size = len(section_paths)
        row_vectors_flat = row_vectors.view(batch_size, -1)  # (batch_size, 32*64)
        col_vectors_flat = col_vectors.view(batch_size, -1)  # (batch_size, 32*64)

        row_features = self.row_linear(row_vectors_flat)  # (batch_size, 128)
        col_features = self.col_linear(col_vectors_flat)  # (batch_size, 128)

        combined_vectors = torch.cat([row_features, col_features], dim=1)  # (batch_size, 256)

        return combined_vectors




# mlflow.set_tracking_uri("http://0.0.0.0:1234")  # MLflow Tracking URI 설정 (필요에 따라 수정)
# run_id = "2656a4c070e2473285171561f6577862"
# model_uri = f"runs:/{run_id}/best_model"  # <run_id>를 실제 run ID로 변경하세요
# model = mlflow.pytorch.load_model(model_uri)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = timm.create_model(model_name=model_name, pretrained=True, num_classes=0)
model = CustomViTRegressor(model_name=base_model)
model = model.to(device)

# model = model.to(device)
# for block in model.base_model.blocks:
#     block.attn.forward = my_forward_wrapper(block.attn)

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
            model = CustomViTRegressor(model_name=model_name).to(device)
            
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
            i=0
            for (images, labels, section_paths) in tqdm(train_loader):
                images, labels = images.to(device), labels.to(device).float()
                
                i+=1
                
                # step (backprop)
                optimizer.zero_grad()
                outputs = model(images, section_paths)
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
                for images, labels, section_paths in val_loader:
                    images, labels = images.to(device), labels.to(device).float()
                    outputs = model(images, section_paths)
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
            if val_loss < best_val_loss:
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

import torch

# 현재 사용 중인 GPU 메모리를 비웁니다.
torch.cuda.empty_cache()

# 현재 예약된 GPU 메모리의 크기를 출력합니다.
print(torch.cuda.memory_reserved())
