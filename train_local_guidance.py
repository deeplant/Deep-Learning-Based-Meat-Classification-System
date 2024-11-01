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
import os

from models.VT_local_gudiance import create_model, TeacherModel
from utils.split_data import split_data
from utils.epoch_local_guidance import regression
from utils.log import log

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

argparser=argparse.ArgumentParser(description='training pipeline')

argparser.add_argument('--experiment', type=str)  # experiment 이름 설정
argparser.add_argument('--run', type=str)  # run 이름 설정
argparser.add_argument('--config', default="./configs/VT_local_guidance.json", type=str)  # model config 파일 경로 설정
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

# 학습할 role 설정
# teacher 설정 시, config 파일 내 ["teacher_model"]["name"] 에 해당 하는 모델을 기본으로 데이터셋을 이용해 fine tuning 한 teacher model 학습 후 저장까지만
# student 설정 시, 저장된 fine tuning 한 teacher model(없다면 기본 ["teacher_model"]["name"] 모델)을 바탕으로 train/test 결과 도출
argparser.add_argument('--role', type=str, choices=['teacher', 'student'], required=True,
                       help='Specify whether to fine-tune the teacher or train the student. Choose between "teacher" or "student".') 

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
distillation_weight = config['hyperparameters'].get('distillation_weight', 0.5)
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

# Initialize model using the custom distillation model
student_model_name = config['models']['model_name']  # Student model name (from the config)
pretrained = config['models']['pretrained']
num_classes = config['models']['num_classes']
in_chans = config['models']['in_chans']
out_dim = config['models']['out_dim']
teacher_model_name = config['models']['teacher_model'].get('name', 'resnet50')
tuned_teacher_model_name = config['models']['teacher_model'].get('finetuned_name', None)
load_from_file = config['models']['teacher_model'].get('load_from_file', False)
teacher_model_path = config['models']['teacher_model'].get('model_path', None)

print('teacher_model_name:::', teacher_model_name)
print('tunedmodel', tuned_teacher_model_name)

# # Create the teacher-student distillation model
# model = create_model(student_model_name, teacher_model_name, pretrained, num_classes, in_chans, out_dim)
# model = model.to(device)

# # Optimizer and scheduler setup
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)

# 학습 파라미터
params_train = {
    'num_epochs':epochs,
    'optimizer':None,
    'train_dl':None,
    'val_dl':None,
    'scheduler':None,
    'save_model':save_model,
    'loss_func': nn.MSELoss(),  # Assuming MSE loss for regression tasks
    'fold':(0, 0),
    'label_names':output_columns,
    'role':args.role,
    'distillation_weight':distillation_weight,
    'tuned_teacher_model_name': tuned_teacher_model_name
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
        mlflow.log_param("distillation_weight", distillation_weight)    


        # train, val 데이터셋
        val_data = fold_data[fold]
        train_data = pd.concat([fold_data[i] for i in range(len(fold_data)) if i != fold])
        val_dataset = MeatDataset(val_data, config, is_train=False)
        train_dataset = MeatDataset(train_data, config, is_train=True)

        # # 모델 만들기
        # model = make_model(config)
        # model = model.to(device)
        
        # Create model with the specified role (teacher or student)
        if args.role == 'teacher':
            # Create teacher model only for fine-tuning
            # Create teacher model, loading from file if specified
            model = TeacherModel(model_name=teacher_model_name, pretrained=pretrained, num_classes=num_classes,
                                 load_from_file=load_from_file, model_path=teacher_model_path)                  
            model = model.to(device)

            # Optimizer for teacher model
            params_train['optimizer'] = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
            



        elif args.role == 'student':
           # Create teacher-student distillation model
            # teacher_model_path = f'./saved_models/{tuned_teacher_model_name}.pth'

            # if os.path.exists(teacher_model_path):
            #     # Load the saved teacher model if it exists
            #     print(f"Loading pre-trained teacher model from {teacher_model_path}")
            #     teacher_model = TeacherModel(model_name=teacher_model_name, pretrained=False, num_classes=num_classes)
            #     teacher_model.load_state_dict(torch.load(teacher_model_path))  # Load trained teacher model
            #     teacher_model = teacher_model.to(device)
            #     teacher_model.eval()  # Ensure teacher model stays frozen during student training
            #     model.teacher_model = teacher_model  # Assign loaded (or new) teacher to the distillation model

            # else:
            #     # Initialize the teacher model from scratch (pretrained ResNet)
            #     print(f"Teacher model not found at {teacher_model_path}, initializing a basic model {teacher_model_name}")

            # Create student-teacher distillation model
            model = create_model(student_model_name, teacher_model_name, pretrained, num_classes, in_chans, out_dim, load_from_file, model_path=teacher_model_path)
            model = model.to(device)

            # Optimizer for student model only (teacher frozen)
            params_train['optimizer'] = optim.Adam(model.student_model.parameters(), lr=lr, weight_decay=weight_decay)

        # 모델의 파라미터 개수 기록
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_params", total_params)

        # # optimizer와 scheduler
        # params_train['optimizer'] = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
        params_train['scheduler'] = scheduler = ReduceLROnPlateau(params_train['optimizer'], mode='min', factor=factor, patience=patience, verbose=True)

        # dataloader
        params_train['train_dl'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
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
