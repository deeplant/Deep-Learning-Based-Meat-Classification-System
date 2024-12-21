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
    def __init__(self, base_model, out_dim, grade_categories):
        super().__init__()
        self.base_model = base_model
        self.out_dim = out_dim

        self.num_features = self.base_model.base_model.num_features  # 768

        # grade를 원핫 벡터 → 16차원 임베딩으로 변환
        self.grade_to_onehot = {grade: idx for idx, grade in enumerate(grade_categories)}
        self.grade_embedding = nn.Sequential(
            nn.Linear(len(grade_categories), 16),  # 5 → 16
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU()
        )

        # 최종 MLP 헤드: (768 + 16) → 1
        self.mlp_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.num_features + 64, 64),  # 768 + 64 → 64
                nn.ReLU(),
                nn.Linear(64, 1)  # 64 → 1
            ) for _ in range(out_dim)
        ])

    def forward(self, x, grade):
        # Base 모델의 출력을 계산
        base_output = self.base_model(x)  # (batch_size, num_features) == (batch_size, 768)

        # grade를 리스트에서 텐서로 변환
        batch_size = len(grade)  # 리스트의 길이를 확인
        grade_onehot = torch.zeros((batch_size, len(self.grade_to_onehot)), device=base_output.device)

        # grade 문자열을 원핫 벡터로 변환
        for i, g in enumerate(grade):
            if g not in self.grade_to_onehot:
                raise ValueError(f"Invalid grade '{g}' encountered. Allowed grades: {list(self.grade_to_onehot.keys())}")
            grade_onehot[i, self.grade_to_onehot[g]] = 1

        # 원핫 벡터 → 16차원 임베딩
        grade_features = self.grade_embedding(grade_onehot)  # (batch_size, 16)

        # 768차원과 16차원 결합
        combined_features = torch.cat([base_output, grade_features], dim=1)  # (batch_size, 768 + 16)

        # 각 MLP 헤드를 통해 계산
        outputs = [head(combined_features) for head in self.mlp_heads]  # list of (batch_size, 1)
        outputs = torch.cat(outputs, dim=1)  # (batch_size, out_dim)

        return outputs

def create_model(model_name, pretrained, num_classes, in_chans, out_dim, config):

    if out_dim <= 0:
        raise ValueError("오류: out_dim이 0 이하입니다.")

    base_model = BaseModel(model_name, pretrained, num_classes, in_chans)
    
    grade_categories = ['등심1++', '등심1+', '등심1', '등심2', '등심3']
    model = MLP_layer(base_model, out_dim=out_dim, grade_categories=grade_categories)

    return model
