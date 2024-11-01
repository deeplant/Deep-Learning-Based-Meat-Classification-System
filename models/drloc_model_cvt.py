import timm
import torch
import torch.nn as nn
from functools import partial
from collections import namedtuple
# from .cvt import ConvolutionalVisionTransformer, LayerNorm, QuickGELU

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(grandparent_dir)

from CvT.CvT_main.lib.models import build_model
from CvT.CvT_main.lib.config.default import _C as default_config, update_config
import yaml
import torch

def load_config():
    config = default_config.clone()
    config_file = 'configs/cvt-13-384x384.yaml'
    class Args:
        cfg = config_file
        opts = []
    update_config(config, Args())
    return config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# n : batch size
# m : number of pairs
# k X k : resolution of the embedding grid
# D : dimension of each token embedding
# x : a tensor of n embedding grids, shape=[n, D, k, k]
def position_sampling(k, m, n):
    pos_1 = torch.randint(k, size=(n, m, 2))
    pos_2 = torch.randint(k, size=(n, m, 2))
    return pos_1.to(device), pos_2.to(device)

def collect_samples(x, pos, n):
    _, c, h, w = x.size()
    x = x.view(n, c, -1).permute(1, 0, 2).reshape(c, -1)
    pos = ((torch.arange(n).long().to(pos.device) * h * w).view(n, 1) + pos[:, :, 0] * h + pos[:, :, 1]).view(-1)
    return (x[:, pos]).view(c, n, -1).permute(1, 0, 2)

def dense_relative_localization_loss(x, drloc_mlp, m):
    n, D, k, k = x.size()
    pos_1, pos_2 = position_sampling(k, m, n)
    deltaxy = abs((pos_1 - pos_2).float())  # [n, m, 2]
    deltaxy /= k
    pts_1 = collect_samples(x, pos_1, n).transpose(1, 2)  # [n, m, D]
    pts_2 = collect_samples(x, pos_2, n).transpose(1, 2)  # [n, m, D]
    predxy = drloc_mlp(torch.cat([pts_1, pts_2], dim=2))
    return nn.L1Loss()(predxy, deltaxy)

class DRLocMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DRLocMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

drloc_mlp = DRLocMLP(2 * 384, 1024, 2).to(device)

class CvtModel(nn.Module):
    def __init__(self):
        super(CvtModel, self).__init__()
        # self.base_model = cvt_model

        config = load_config()

        self.base_model = build_model(config)

        pretrained_model_path = getattr(config.TEST, 'MODEL_FILE', None)
        if pretrained_model_path:
            print(f'Loading pretrained model from {pretrained_model_path}')
            state_dict = torch.load(pretrained_model_path, map_location='cpu')
            missing_keys, unexpected_keys = self.base_model.load_state_dict(state_dict, strict=False)
                
            # 가중치 로딩 결과 확인
            if missing_keys:
                print(f'Missing keys when loading pretrained weights: {missing_keys}')
            if unexpected_keys:
                print(f'Unexpected keys when loading pretrained weights: {unexpected_keys}')
            if not missing_keys and not unexpected_keys:
                print('Pretrained weights loaded successfully.')
        else:
            print("no pretrained model")

        self.base_model.head = nn.Identity()

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, m):
        
        x, _ = self.base_model.stage0(x)
        
        x, _ = self.base_model.stage1(x)
        
        x, cls_tokens = self.base_model.stage2(x) # x : n, d, k, k

        x = self.avg_pool(x) # 14x14에서 7x7로 다운샘플링

        # drloc_loss 계산
        drloc_loss = dense_relative_localization_loss(x, drloc_mlp, m)
        
        cls_tokens = self.base_model.norm(cls_tokens)
        cls_tokens = torch.squeeze(cls_tokens, dim=1)
        
        return cls_tokens, drloc_loss

class MLP_layer(nn.Module):
    def __init__(self, base_model, out_dim):
        super().__init__()
        self.base_model = base_model
        self.out_dim = out_dim

        self.num_features = self.base_model.base_model.num_features
        
        # out_dim 개수만큼 MLP 생성
        self.mlp_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.num_features, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(out_dim)
        ])

    def forward(self, x, m):
        base_output, drloc_loss = self.base_model(x, m)
        #base_output = self.base_model(x)
        
        features = base_output

        outputs = [head(features) for head in self.mlp_heads]

        return torch.cat(outputs, dim=1), drloc_loss

def create_model(model_name, pretrained, num_classes, in_chans, out_dim):

    if out_dim <= 0:
        raise ValueError("오류: out_dim이 0 이하입니다.")

    base_model = CvtModel()
    model = MLP_layer(base_model, out_dim)

    return model
