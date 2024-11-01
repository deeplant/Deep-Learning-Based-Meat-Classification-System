#custom model

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# !export PYTHONPATH=$PYTHONPATH:/home/work/tiny-transformers/tiny-transformers

# +
# from pycls.models.cnns.resnet import ResNet  # resnet.py에서 ResNet 모델 불러오기

class TeacherModel(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=0, load_from_file=False, model_path=None):
        super(TeacherModel, self).__init__()
        self.model = None
        self.features = []
        self.feature_layers = ['layer1', 'layer2', 'layer3']

        if load_from_file and model_path:
            # Load model from a saved .pth file
            self._load_pretrained_model(model_path)
        else:
            # Use timm to create a model
            self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        
        self.model.eval()
        self._register_hooks()

    def _load_pretrained_model(self, model_path):
        # Load the saved model from the provided path
        print(f"Loading teacher model from {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        # self.model.load_state_dict(checkpoint, strict=False)
        # self.model = nn.Sequential(*list(checkpoint['model'].children())[:-1])  # Assuming last two layers are not required
        # self.model.load_state_dict(checkpoint['state_dict'])
        # Check if the checkpoint is a full model or just a state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']  # If saved as a state dict
        else:
            state_dict = checkpoint  # If it's directly the state dict

        # Initialize timm model
        self.model = timm.create_model(model_name='resnet50', pretrained=False, num_classes=0)
        
        # Adjust the state dict keys if necessary (remove 'base_model.base_model.')
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('base_model.base_model.', 'model.')  # Adjust the prefix
            new_state_dict[new_key] = state_dict[key]

        # Load state dict into model
        self.model.load_state_dict(new_state_dict, strict=False)
        print("Teacher model loaded successfully!")

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
        self.feature_layers = ['layers.0', 'layers.1', 'layers.2', 'layers.3']  # Hook SwinTransformerStage blocks
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
        # 입력이 [B, H, W, C] 형태라면 [B, C, H, W]로 변환
        if x.dim() == 4 and x.shape[1] != x.shape[-1]:
            x = x.permute(0, 3, 1, 2)
        
        x = self.conv(x)
        
        
        return x


class DistillationModel(nn.Module):
    def __init__(self, teacher_model, student_model):
        super(DistillationModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.output_transform = nn.Linear(768, 5)
        self.transforms = None  # 초기에는 None으로 설정

    def forward(self, x):
        # print("DistillationModel forward pass started", flush=True)

        device = x.device  # Get the device of the input tensor

        with torch.no_grad():
            teacher_output, teacher_features = self.teacher_model(x)  # Teacher features remain frozen.
    
        student_output, student_features = self.student_model(x)  # Student features should be updated.

        # transforms가 아직 초기화되지 않았다면 여기서 초기화
        if self.transforms is None:
            self.transforms = nn.ModuleList()
            for t_feat, s_feat in zip(teacher_features, student_features[:-1]):
                in_channels = s_feat.shape[-1]  # Student feature 채널 수
                out_channels = t_feat.shape[1]  # Teacher feature 채널 수
                transform = FeatureTransform(in_channels, out_channels)
                self.transforms.append(transform.to(device))  # Move the transform to the correct device


        # 손실 및 변환을 동시에 계산
        distillation_loss = torch.tensor(0.0, device=device, requires_grad=True)  # 초기화
        for i, (transform, s_feat, t_feat) in enumerate(zip(self.transforms, student_features[:-1], teacher_features)):
            # print(f"Layer {i+1} - Before Transformation - Student feature shape: {s_feat.shape}")
            # print(f"Before Conv: {s_feat.shape}", flush=True)
            
            # 학생 특징 변환
            transformed_feat = transform(s_feat)  # permute 필요 없음
            # print(f"Layer {i+1} - After Transformation - Transformed feat shape: {transformed_feat.shape}", flush=True)

            # 손실 계산
            layer_loss = F.mse_loss(transformed_feat, t_feat)
            distillation_loss = distillation_loss + layer_loss
            # print(f"Layer {i+1} loss: {layer_loss.item()}", flush=True)

        student_output_resized = self.output_transform(student_output)
        return student_output_resized, teacher_output, distillation_loss



# # MLP Layer to output 'out_dim' different outputs
# class MLP_layer(nn.Module):
#     def __init__(self, base_model, out_dim):
#         super().__init__()
#         self.base_model = base_model
#         self.out_dim = out_dim

#         # Get the number of features from the base model
#         self.num_features = self.base_model.base_model.num_features

#         # Create MLP heads, one for each output dimension (out_dim = 5 in this case)
#         self.mlp_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.num_features, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 1)
#             ) for _ in range(out_dim)
#         ])

#     def forward(self, x):
#         base_output = self.base_model(x)
#         features = base_output

#         # Apply each MLP head to the base model output
#         outputs = [head(features) for head in self.mlp_heads]

#         # Concatenate the outputs along the last dimension
#         return torch.cat(outputs, dim=1)

def create_model(student_model_name, teacher_model_name, pretrained, num_classes, in_chans, out_dim, load_from_file=False, model_path=None):
    teacher_model = TeacherModel(model_name=teacher_model_name, pretrained=pretrained, num_classes=num_classes, load_from_file=load_from_file, model_path=model_path)
    student_model = StudentModel(model_name=student_model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
    model = DistillationModel(teacher_model, student_model)

    return model
