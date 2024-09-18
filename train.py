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
from utils.epoch import regression
from utils.log import log

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

argparser=argparse.ArgumentParser(description='training pipeline')

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
argparser.add_argument('--port', type=int) # 포트 설정

args=argparser.parse_args()

# config 파일을 읽고 기본 값 설정
with open(args.config, 'r') as json_file:
    config = json.load(json_file)

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
port = args.port if args.port is not None else 5000

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


##########################################################################################################################################
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

# 학습 파라미터
params_train = {
    'num_epochs':epochs,
    'optimizer':None,
    'train_dl':None,
    'val_dl':None,
    'scheduler':None,
    'save_model':save_model,
    'loss_func':None,
    'fold':(0, 0),
    'label_names':output_columns
}

all_fold_results = []
for fold in range(n_folds):
    # cross_validation 시 run name 뒤에 fold (숫자) 붙임
    if cross_validation:
        r = run_name + " fold " + str(fold+1)
    else:
        r = run_name

    with mlflow.start_run(run_name=r) as run:
        print(f"run_id: {run.info.run_id}")

        # 기본 파라미터 mlflow에 기록
        mlflow.log_dict(config, 'config/configs.json')
        mlflow.log_param("model", config["models"].get('model_name', 'null'))
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("weight_decay", weight_decay)    

        # train, val 데이터셋
        val_data = fold_data[fold]
        train_data = pd.concat([fold_data[i] for i in range(len(fold_data)) if i != fold])
        val_dataset = MeatDataset(val_data, config, is_train=False)
        train_dataset = MeatDataset(train_data, config, is_train=True)

        # 모델 만들기
        model = make_model(config)
        model = model.to(device)

        # 모델의 파라미터 개수 기록
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_params", total_params)

        # optimizer와 scheduler
        params_train['optimizer'] = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
        params_train['scheduler'] = scheduler = ReduceLROnPlateau(params_train['optimizer'], mode='min', factor=factor, patience=patience, verbose=True)

        # dataloader
        params_train['train_dl'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, seed = seed)
        params_train['val_dl'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        ## loss function
        params_train['loss_func'] = nn.MSELoss()

        if cross_validation:
            params_train['fold'] = (fold + 1, cross_validation)

        # 학습 진행
        fold_result = regression(model, params_train)

        # fold 결과 저장
        all_fold_results.append(fold_result)

        # 마지막 fold일 경우 결과 기록
        if fold + 1 == n_folds:
            log(all_fold_results, cross_validation, params_train['label_names'])