import torch
import timm  # Assuming you use timm to create models
import torch.nn as nn

# Define the same model class as the one used to save the .pth file
class TeacherModel(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=0):
        super(TeacherModel, self).__init__()
        # Load the model using timm (ResNet)
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

# 1. Load the model from timm
timm_model = TeacherModel(model_name='resnet50', pretrained=False)

# Print the timm model architecture
print("Timm Model Architecture:\n")
print(timm_model)

# Print the state_dict keys from the timm model
print("\nTimm Model State Dict Keys:\n")
print(timm_model.state_dict().keys())

# 2. Load the saved model state dictionary from the fine-tuned .pth file
checkpoint_path = './saved_models/fine_tuned_model_epoch20_fold0.pth'  # Update with your path
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Print the checkpoint's keys (this will help see what structure is in the file)
print("\nFine-tuned Model Checkpoint Keys:\n")
print(checkpoint.keys())  # This will print top-level keys in the checkpoint

# Check if it's a state_dict or a full model
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Print the keys in the state_dict of the fine-tuned model
print("\nFine-tuned Model State Dict Keys:\n")
print(state_dict.keys())

# 3. Initialize the timm-based ResNet architecture (to match the fine-tuned model)
fine_tuned_model = TeacherModel(model_name='resnet50', pretrained=False)

# Load the state dictionary into the model (with strict=False to allow flexibility)
try:
    fine_tuned_model.load_state_dict(state_dict, strict=False)
    print("\nFine-tuned model loaded successfully!\n")
except RuntimeError as e:
    print(f"\nError loading fine-tuned model: {e}\n")

# 4. Compare specific layer weights (for debugging)
print("\nComparing Weights Between Timm Model and Fine-tuned Model:\n")

# Example: Compare the first convolutional layer weights (conv1)
print("Timm Model Conv1 Weights:\n")
print(timm_model.model.conv1.weight)

print("\nFine-tuned Model Conv1 Weights:\n")
print(fine_tuned_model.model.conv1.weight)
