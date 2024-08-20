import timm
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, in_chans):
        super(BaseModel, self).__init__()
        self.base_model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    def forward(self, x):
        x = self.base_model(x)
        return x
    
    

class MLP_layer(nn.Module):
    def __init__(self, base_model, out_dim):
        super().__init__()
        self.base_model = base_model
        self.out_dim = out_dim

        if out_dim > 0:
            self.num_features = self.base_model.num_features
            
            # out_dim 개수만큼 MLP 생성
            self.mlp_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.num_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                ) for _ in range(out_dim)
            ])
        else:
            self.mlp_heads = None

    def forward(self, x):
        base_output = self.base_model(x)
        
        if self.out_dim > 0:
            features = base_output

            outputs = [head(features) for head in self.mlp_heads]
            return torch.cat(outputs, dim=1)
        else:
            
            return base_output
    

def create_model(model_name, pretrained, num_classes, in_chans, out_dim):

    if out_dim <= 0:
        print("분류 작업 진행: out_dim이 0 이하입니다.")

    base_model = BaseModel(model_name, pretrained, num_classes, in_chans)
    model = MLP_layer(base_model, out_dim)

    return model
