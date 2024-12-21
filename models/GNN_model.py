import timm
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# +

class BaseModel(nn.Module):
    def __init__(self, model_name, pretrained, gnn_out_channels, num_classes, in_chans):
        super(BaseModel, self).__init__()
        
        # 1. CNN 모델을 사용해 이미지 특징 추출 (기존 코드 사용)
        self.cnn_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans)
        self.num_features = self.cnn_model.num_features
        
        # 2. GNN 모델 정의 (예: GCN)
        self.gnn_model = GCNConv(self.num_features, gnn_out_channels)
        
        # 최종 분류 레이어
        self.fc = nn.Linear(gnn_out_channels, num_classes)

    def forward(self, images, edge_index):
        # CNN을 통해 이미지 특징 추출
        x = self.cnn_model(images)
        
        # GNN에 전달 (노드 특징과 엣지 인덱스)
        x = self.gnn_model(x, edge_index)
        x = self.fc(x)
        return x

# -

    

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

    def forward(self, x):
        base_output = self.base_model(x)
        
        features = base_output

        outputs = [head(features) for head in self.mlp_heads]

        return torch.cat(outputs, dim=1)

def create_model(model_name, pretrained, num_classes, in_chans, out_dim, config):

    if out_dim <= 0:
        raise ValueError("오류: out_dim이 0 이하입니다.")

    base_model = BaseModel(model_name, pretrained, num_classes, in_chans)
    model = MLP_layer(base_model, out_dim)

    return model


