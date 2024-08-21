# 학습 코드 작성

import argparse
import json
import torch
import pandas as pd
from utils.split_data import split_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

argparser=argparse.ArgumentParser(description='training pipeline')

argparser.add_argument('--experiment', default ='test', type=str)  # experiment 이름 설정
argparser.add_argument('--run', default ='test', type=str)  # run 이름 설정
argparser.add_argument('--config', default="./configs/default_vit_config.json", type=str)  # model config 파일 경로 설정
argparser.add_argument('--save_model', action='store_true')  # 모델 저장
argparser.add_argument('--epochs', type=int)  #epochs
argparser.add_argument('--lr', '--learning_rate', type=float)  # learning rate
argparser.add_argument('--batch_size', type=int)
argparser.add_argument('--num_workers', type=int)
argparser.add_argument('--csv_path', default='./dataset/labels/default.csv')
argparser.add_argument('--seed', type=int)
argparser.add_argument('--cross_validation', type=int)

args=argparser.parse_args()

with open(args.config, 'r') as json_file:
    config = json.load(json_file)

args.experiment = args.experiment if args.experiment is not False else config.get('experiment', 'test')
args.run = args.run if args.run is not False else config.get('run', 'test')
args.save_model = args.save_model if args.save_model is not False else config['models'].get('save_model', False)
args.epochs = args.epochs if args.epochs is not None else config['hyperparameters'].get('epochs', 20)
args.lr = args.lr if args.lr is not None else config['hyperparameters'].get('learning_rate', 1e-5)
args.batch_size = args.batch_size if args.batch_size is not None else config['hyperparameters'].get('batch_size', 32)
args.num_workers = args.num_workers if args.num_workers is not None else config['hyperparameters'].get('num_workers', 4)
args.seed = args.seed if args.seed is not None else config['hyperparameters'].get('seed', 42)
args.cross_validation = args.cross_validation if args.cross_validation is not None else config['hyperparameters'].get('cross_validation', 0)

experiment = args.experiment
run = args.run
save_model = args.save_model
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
num_workers = args.num_workers
seed = args.seed
cross_validation = args.cross_validation

csv = pd.read_csv(args.csv_path)
output_columns = config['output_columns']
print(output_columns)

fold_data = split_data(csv, output_columns, cross_validation)




# print(f"Run name: {args.run}")
# print(f"Experiment name: {args.experiment}")
# print(f"Save model: {args.save_model}")
# print(f"Number of epochs: {args.epochs}")
# print(f"Learning rate: {args.lr}")
# print(f"Batch size: {args.batch_size}")
# print(f"Number of workers: {args.num_workers}")