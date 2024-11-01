import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, in_chans):
        super(BaseModel, self).__init__()
        self.base_model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    def forward(self, x):
        x = self.base_model(x)
        return x

class LGuidanceLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LGuidanceLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.guidance = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = self.fc(x)
        return x * self.guidance

class MLP_layer(nn.Module):
    def __init__(self, base_model, out_dim):
        super().__init__()
        self.base_model = base_model
        self.out_dim = out_dim

        self.num_features = self.base_model.base_model.num_features
        
        self.l_guidance = LGuidanceLayer(self.num_features, 128)
        
        self.mlp_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(out_dim)
        ])

    def forward(self, x):
        base_output = self.base_model(x)
        
        features = self.l_guidance(base_output)

        outputs = [head(features) for head in self.mlp_heads]

        return torch.cat(outputs, dim=1)

    def l_guidance_loss(self, outputs, targets, lambda_param=0.01):
        mse_loss = F.mse_loss(outputs, targets)
        guidance_norm = torch.norm(self.l_guidance.guidance)
        return mse_loss + lambda_param * guidance_norm

def create_model(model_name, pretrained, num_classes, in_chans, out_dim):
    if out_dim <= 0:
        raise ValueError("오류: out_dim이 0 이하입니다.")

    base_model = BaseModel(model_name, pretrained, num_classes, in_chans)
    model = MLP_layer(base_model, out_dim)

    return model
