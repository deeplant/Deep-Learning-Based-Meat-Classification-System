import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow

# !export PYTHONPATH=$PYTHONPATH:/home/work/tiny-transformers/tiny-transformers

# +
from pycls.models.cnns.resnet import ResNet  # resnet.py에서 ResNet 모델 불러오기

class TeacherModel(nn.Module):
    def __init__(self, resnet_depth=56, num_classes=100):
        super(TeacherModel, self).__init__()
        self.model = ResNet(depth=resnet_depth, num_classes=num_classes)
        self.model.eval()
        self.features = []
        self.feature_layers = ['layer1', 'layer2', 'layer3']
        self._register_hooks()

    def _register_hooks(self):
        def hook(module, input, output):
            self.features.append(output)
        
        for name, module in self.model.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(hook)

    def forward(self, x):
        self.features = []
        output = self.model(x)
        return output, self.features



# -

# MLflow 설정 및 모델 로드
mlflow.set_tracking_uri("http://0.0.0.0:5000")
run_id = "f361150c5bc348bfa4c71ef0904e4003"
model_uri = f"runs:/{run_id}/best_model"

class TeacherModel(nn.Module):
    def __init__(self, model_uri):
        super(TeacherModel, self).__init__()
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.eval()
        self.features = []
        self.feature_layers = ['layer1', 'layer2', 'layer3']
        self._register_hooks()

    def _register_hooks(self):
        def hook(module, input, output):
            self.features.append(output)
        
        for name, module in self.model.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(hook)

    def forward(self, x):
        self.features = []
        output = self.model(x)
        return output, self.features

class StudentModel(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, in_chans):
        super(StudentModel, self).__init__()
        self.base_model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
        self.features = []
        self.feature_layers = ['layer1', 'layer2', 'layer3']
        self._register_hooks()

    def _register_hooks(self):
        def hook(module, input, output):
            self.features.append(output)
        
        for name, module in self.base_model.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(hook)

    def forward(self, x):
        self.features = []
        output = self.base_model(x)
        return output, self.features

class FeatureTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureTransform, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DistillationModel(nn.Module):
    def __init__(self, teacher_model, student_model):
        super(DistillationModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.transforms = nn.ModuleList()

        # 특징 변환 레이어 초기화
        for t_feat, s_feat in zip(self.teacher_model.features, self.student_model.features):
            self.transforms.append(FeatureTransform(s_feat.shape[1], t_feat.shape[1]))
        
        # 학생 모델의 출력을 5개로 조정하는 레이어 추가
        self.output_transform = nn.Linear(768, 5)

    def forward(self, x):
        with torch.no_grad():
            teacher_output, teacher_features = self.teacher_model(x)
        
        student_output, student_features = self.student_model(x)
        
        transformed_features = []
        for transform, feat in zip(self.transforms, student_features):
            transformed_features.append(transform(feat))
        
        # 학생 모델의 출력을 5개로 조정
        student_output_resized = self.output_transform(student_output)

        return student_output_resized, teacher_output, transformed_features, teacher_features


def create_model(student_model_name, pretrained, num_classes, in_chans,out_dim):
    teacher_model = TeacherModel(model_uri)
    student_model = StudentModel(student_model_name, pretrained, num_classes, in_chans)
    model = DistillationModel(teacher_model, student_model)

    return model
