import timm
import torch
import torch.nn as nn
import math


def conv3x3(in_dim, out_dim):
    return torch.nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_dim)
    )

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, dim, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, dim, 1, 1]))

    def forward(self, x):
        return x * self.alpha + self.beta

class SOPE(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.pre_affine = Affine(3)
        self.post_affine = Affine(embed_dim)

        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 8),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim),
            )
        elif patch_size[0] == 4:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim),
            )
        elif patch_size[0] == 2:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim),
                nn.GELU(),
            )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pre_affine(x)
        x = self.proj(x)
        x = self.post_affine(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x


class DAFF(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hid_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, kernel_size=3, stride=1, padding=(kernel_size-1)//2, groups=hid_dim)
        self.conv3 = nn.Conv2d(hid_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Linear(in_dim, in_dim // 4)
        self.excitation = nn.Linear(in_dim // 4, in_dim)
        self.bn1 = nn.BatchNorm2d(hid_dim)
        self.bn2 = nn.BatchNorm2d(hid_dim)
        self.bn3 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        B, N, C = x.size()
        cls_token, tokens = torch.split(x, [1, N-1], dim=1)
        x = tokens.reshape(B, int(math.sqrt(N-1)), int(math.sqrt(N-1)), C).permute(0, 3, 1, 2)
        x = self.act(self.bn1(self.conv1(x)))
        x = x + self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        weight = self.squeeze(x).flatten(1).reshape(B, 1, C)
        weight = self.excitation(self.act(self.compress(weight)))
        cls_token = cls_token * weight
        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)
        return out



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.ht_proj = nn.Linear(head_dim, dim, bias=True)
        self.ht_norm = nn.LayerNorm(head_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_heads, dim))
        self.attn_drop = nn.Dropout(p=0.0)  # 원래 모델과 일치하도록 추가

    def forward(self, x):
        B, N, C = x.shape
        # head token
        head_pos = self.pos_embed.expand(x.shape[0], -1, -1)
        ht = x.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        ht = ht.mean(dim=2)
        ht = self.ht_proj(ht).reshape(B, -1, self.num_heads, C // self.num_heads)
        ht = self.act(self.ht_norm(ht)).flatten(2)
        ht = ht + head_pos
        x = torch.cat([x, ht], dim=1)
        # common MHSA
        qkv = self.qkv(x).reshape(B, N + self.num_heads, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N + self.num_heads, C)
        x = self.proj(x)
        # split, average and add
        cls, patch, ht = torch.split(x, [1, N-1, self.num_heads], dim=1)
        cls = cls + torch.mean(ht, dim=1, keepdim=True)
        x = torch.cat([cls, patch], dim=1)
        return x

class BaseModel(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, in_chans, embed_layer=SOPE):
        super(BaseModel, self).__init__()
        self.base_model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
        
        # SOPE
        self.base_model.patch_embed = SOPE(patch_size=(16, 16), embed_dim=768)
        
        for i in range(len(self.base_model.blocks)):
            block = self.base_model.blocks[i]

            # HI-MHSA
            old_attn = block.attn
            dim = old_attn.qkv.in_features
            num_heads = old_attn.num_heads

            block.attn = Attention(dim, num_heads)

            # DAFF
            in_features = block.mlp.fc1.in_features
            hidden_features = block.mlp.fc1.out_features
            out_features = block.mlp.fc2.out_features

            block.mlp = DAFF(in_features, hidden_features, out_features)
        
    def forward(self, x):
        x = self.base_model(x)
        return x

    

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
