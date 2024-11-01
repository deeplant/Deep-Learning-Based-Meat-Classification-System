import argparse
import json
import pandas as pd
import mlflow
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.dataset import MeatDataset
from torch.utils.data import DataLoader
import random
import numpy as np

from models.model import make_model
from utils.split_data import split_data
from utils.epoch_with_lg import regression, regression_with_local_guidance
from utils.log import log
from torchvision.models import resnet56  # ResNet56 사용
import random
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Argument parser 설정
argparser = argparse.ArgumentParser(description='training pipeline with Local Guidance')

argparser.add_argument('--experiment', type=str)  # experiment 이름 설정
argparser.add_argument('--run', type=str)  # run 이름 설정
argparser.add_argument('--config', default="./configs/default_ViT_config.json", type=str)  # model config 파일 경로 설정
argparser.add_argument('--save_model', action='store_true')  # 모델 저장
argparser.add_argument('--epochs', type=int)  #epochs
argparser.add_argument('--lr', '--learning_rate', type=float)  # learning rate
argparser.add_argument('--batch_size', type=int)
argparser.add_argument('--weight_decay', type=float) # Adam의 weight_decay
argparser.add_argument('--num_workers', type=int) # 서브 프로세스 개수 (cpu 코어 개수) - 기본 4개
argparser.add_argument('--csv_path', default='./dataset/default.csv') # csv 경로
argparser.add_argument('--seed', type=int) # 랜덤 시드
argparser.add_argument('--cross_validation', type=int) # k-fold-cross-validation. 0: 비활성화, 2이상: k-fold 활성화 및 fold 개수 지정
argparser.add_argument('--port', default=5000, type=int) # 포트 설정
argparser.add_argument('--mode', choices=['teacher', 'student'], required=True)  # 추가된 옵션

args=argparser.parse_args()

# config 파일을 읽고 기본 값 설정
with open(args.config, 'r') as json_file:
    config = json.load(json_file)
    
# 기본 변수 설정
experiment = args.experiment if args.experiment is not None else config.get('experiment', 'test')
run_name = args.run if args.run is not None else config.get('run', 'test')
save_model = args.save_model if args.save_model is not False else config['models'].get('save_model', False)
epochs = args.epochs if args.epochs is not None else config['hyperparameters'].get('epochs', 20)
lr = args.lr if args.lr is not None else config['hyperparameters'].get('learning_rate', 1e-5)
batch_size = args.batch_size if args.batch_size is not None else config['hyperparameters'].get('batch_size', 32)
weight_decay = args.weight_decay if args.weight_decay is not None else config['hyperparameters'].get('weight_decay', 5e-4)
num_workers = args.num_workers if args.num_workers is not None else config['hyperparameters'].get('num_workers', 4)
seed = args.seed if args.seed is not None else config['hyperparameters'].get('seed', 42)
cross_validation = args.cross_validation if args.cross_validation is not None else config['hyperparameters'].get('cross_validation', 0)
factor = config['hyperparameters'].get('factor', 0.3)
patience = config['hyperparameters'].get('patience', 2)
port = args.port

# 랜덤 시드 설정
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# csv 파일 읽기
csv = pd.read_csv(args.csv_path)
output_columns = config['output_columns']
print(output_columns)

# 데이터 분할 (cross_validation의 수 만큼 데이터 분할, cross_validation=0이면 5개로 분할 -> train 4 : val 1)
fold_data = split_data(csv, output_columns, cross_validation)


# #########################################################################################################################################
# train

# mlflow 설정
mlflow.set_tracking_uri('http://0.0.0.0:'+str(port))
mlflow.set_experiment(experiment)

if cross_validation:
    n_folds = cross_validation
    print("\n" + "="*50)
    print(f"     K-Fold Cross Validation (K={n_folds}) Enabled")
    print("="*50 + "\n")
else:
    n_folds = 1
    print("\n" + "="*50)
    print("     Single Fold Training")
    print("="*50 + "\n")

# 학습 단계 설정
if args.mode == 'teacher':
    # CNN (ResNet56) 학습 단계
    print("Training teacher (CNN ResNet56)")
    teacher_model = resnet56(pretrained=False, num_classes=len(output_columns))
    teacher_model = teacher_model.to(device)
    
    params_train = {
        'num_epochs': epochs,
        'optimizer': optim.Adam(teacher_model.parameters(), lr=lr, weight_decay=weight_decay),
        'scheduler': ReduceLROnPlateau(optim.Adam(teacher_model.parameters(), lr=lr, weight_decay=weight_decay), mode='min', factor=factor, patience=patience, verbose=True),
        'train_dl': None,
        'val_dl': None,
        'save_model': save_model,
        'loss_func': nn.MSELoss(),
        'fold': (0, 0),
        'label_names': output_columns
    }
    
    for fold in range(n_folds):
        with mlflow.start_run(run_name=f'{run_name}_teacher_fold_{fold + 1}'):
            # 데이터를 fold에 맞게 설정
            train_data = pd.concat([fold_data[i] for i in range(len(fold_data)) if i != fold])
            val_data = fold_data[fold]
            train_dataset = MeatDataset(train_data, config, is_train=True)
            val_dataset = MeatDataset(val_data, config, is_train=False)

            params_train['train_dl'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            params_train['val_dl'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            regression(teacher_model, params_train)

elif args.mode == 'student':
    # VT 학습 단계 (CNN이 학습한 지식 활용)
    print("Training student (Vision Transformer)")
    teacher_model = resnet56(pretrained=False, num_classes=len(output_columns))  # Teacher 모델을 불러옴
    teacher_model.load_state_dict(torch.load('./path_to_saved_teacher_model.pth'))  # 학습된 teacher 모델 로드
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # 학습된 모델은 평가 모드로 설정

    student_model = make_model(config)
    student_model = student_model.to(device)

    # Teacher와 함께 학습
    params_train = {
        'num_epochs': epochs,
        'optimizer': optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay),
        'scheduler': ReduceLROnPlateau(optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay), mode='min', factor=factor, patience=patience, verbose=True),
        'train_dl': None,
        'val_dl': None,
        'save_model': save_model,
        'loss_func': nn.MSELoss(),
        'fold': (0, 0),
        'label_names': output_columns
    }

    for fold in range(n_folds):
        with mlflow.start_run(run_name=f'{run_name}_student_fold_{fold + 1}'):
            train_data = pd.concat([fold_data[i] for i in range(len(fold_data)) if i != fold])
            val_data = fold_data[fold]
            train_dataset = MeatDataset(train_data, config, is_train=True)
            val_dataset = MeatDataset(val_data, config, is_train=False)

            params_train['train_dl'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            params_train['val_dl'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

            # Local Guidance (CNN 피처와 VT 피처의 일치)
            regression_with_local_guidance(student_model, teacher_model, params_train)
