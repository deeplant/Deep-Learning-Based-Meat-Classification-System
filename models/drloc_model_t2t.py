import timm
import torch
import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(grandparent_dir)


from T2T_ViT.T2T_ViT_main.models.t2t_vit import *
from T2T_ViT.T2T_ViT_main.utils import load_for_transfer_learning 


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


class BaseModel(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, in_chans):
        super(BaseModel, self).__init__()
        self.base_model = t2t_vit_14()
        load_for_transfer_learning(self.base_model, 'models/weights/81.5_T2T_ViT_14.pth', use_ema=True, strict=False, num_classes=0)

        self.base_model.head = nn.Identity()
        
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x, m):
        B = x.shape[0]
        
        x = self.base_model.tokens_to_token(x)
        
        cls_tokens = self.base_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.base_model.pos_embed
        x = self.base_model.pos_drop(x)
        
        for i, block in enumerate(self.base_model.blocks) :
            x = block(x)
        
        x = self.base_model.norm(x)
        
        cls_token = x[:, 0]
        
        patch_token = x[:, 1:] # (batch_size, 196, 768)
        batch_size, num_patches, embed_dim = patch_token.shape
        patch_token = patch_token.view(batch_size, 14, 14, embed_dim) # n, k, k, d
        patch_token = patch_token.permute(0, 3, 1, 2) # n, d, k, k
        patch_token = self.avg_pool(patch_token) # n, d, 14, 14 -> n, d, 7, 7
        
        drloc_loss = dense_relative_localization_loss(patch_token, drloc_mlp, m)
        
        
        return cls_token, drloc_loss

    

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
        
        features = base_output

        outputs = [head(features) for head in self.mlp_heads]

        return torch.cat(outputs, dim=1), drloc_loss

def create_model(model_name, pretrained, num_classes, in_chans, out_dim):

    if out_dim <= 0:
        raise ValueError("오류: out_dim이 0 이하입니다.")

    base_model = BaseModel(model_name, pretrained, num_classes, in_chans)
    model = MLP_layer(base_model, out_dim)

    return model
