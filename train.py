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

argparser.add_argument('--experiment', default ='test', type=str)  # experiment 이름 설정
argparser.add_argument('--run', default ='test', type=str)  # run 이름 설정
argparser.add_argument('--config', default="./configs/default_ViT_config.json", type=str)  # model config 파일 경로 설정
argparser.add_argument('--save_model', action='store_true')  # 모델 저장
argparser.add_argument('--epochs', type=int)  #epochs
argparser.add_argument('--lr', '--learning_rate', type=float)  # learning rate
argparser.add_argument('--batch_size', type=int)
argparser.add_argument('--weight_decay', type=float)
argparser.add_argument('--num_workers', type=int)
argparser.add_argument('--csv_path', default='./dataset/default.csv')
argparser.add_argument('--seed', type=int)
argparser.add_argument('--cross_validation', type=int)

args=argparser.parse_args()

with open(args.config, 'r') as json_file:
    config = json.load(json_file)

experiment = args.experiment if args.experiment is not False else config.get('experiment', 'test')
run_name = args.run if args.run is not False else config.get('run', 'test')
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

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

csv = pd.read_csv(args.csv_path)
output_columns = config['output_columns']
print(output_columns)

fold_data = split_data(csv, output_columns, cross_validation)


##########################################################################################################################################
# train

# mlflow 설정
mlflow.set_tracking_uri('http://0.0.0.0:포트')
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
    if cross_validation:
        r = run_name + " fold " + str(fold+1)
    else:
        r = run_name

    with mlflow.start_run(run_name=r) as run:
        print(run.info.run_id)
        mlflow.log_dict(config, 'config/configs.json')
        mlflow.log_param("model", config["models"].get('model_name', 'null'))
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("weight_decay", weight_decay)    
    
        val_data = fold_data[fold]
        train_data = pd.concat([fold_data[i] for i in range(len(fold_data)) if i != fold])
        val_dataset = MeatDataset(val_data, config, is_train=False)
        train_dataset = MeatDataset(train_data, config, is_train=True)

        model = make_model(config)
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_params", total_params)

        params_train['optimizer'] = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
        params_train['scheduler'] = scheduler = ReduceLROnPlateau(params_train['optimizer'], mode='min', factor=factor, patience=patience, verbose=True)
        params_train['train_dl'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, seed = seed)
        params_train['val_dl'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


        ## Regression
        params_train['loss_func'] = nn.MSELoss()

        if cross_validation:
            params_train['fold'] = (fold + 1, cross_validation)

        fold_result = regression(model, params_train)

        all_fold_results.append(fold_result)

        if fold + 1 == n_folds:
            log(all_fold_results, cross_validation, params_train['label_names'])