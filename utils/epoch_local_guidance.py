# train, val 1 epoch를 진행하는 함수
# train 모드에서는 train과 validation 진행
# evaluate 모드에서는 validation만 진행
# role에 따라 다른 방식으로 작동.

from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import r2_score
import mlflow
import torch.nn.functional as F


# Dual task loss: combines task loss and distillation loss
def dual_task_loss(vt_output, teacher_output, distillation_loss, labels, beta=2.5):
    cls_loss = F.mse_loss(vt_output, labels)  # Classification loss
    # l2_loss = F.mse_loss(vt_features, teacher_features)  # Distillation loss
    return cls_loss + beta * distillation_loss


def regression(model, params):
    num_epochs = params['num_epochs']
    optimizer = params['optimizer']
    train_loader = params['train_dl']
    val_loader = params['val_dl']
    scheduler = params['scheduler']
    save_model = params['save_model']
    loss_func = params['loss_func']
    fold, n_folds = params['fold']
    label_names = params['label_names']
    role = params['role']
    distillation_weight = params['distillation_weight']  # Weight for distillation loss
    tuned_teacher_model_name = params['tuned_teacher_model_name']

    print("save model:", save_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_r2 = []
    val_r2 = []
    val_acc = []
    average_val_acc = []

    for epoch in range(num_epochs):
        print(f"\n{'-'*15} Epoch {epoch+1}/{num_epochs} {'-'*15}", end='')
        if fold:
            print(f" (Fold {fold}/{n_folds})\n")
        else:
            print()

        ###### Train ######
        model.train()
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()

            if role == 'teacher':
                # Only forward pass through the teacher model
                teacher_output, teacher_features = model(images)
                task_loss = loss_func(teacher_output, labels)
                total_loss = task_loss 
                # student_output = None


            elif role == 'student':
                # Forward pass through distillation model (returns student output and distillation loss)
                student_output, teacher_output, distillation_loss = model(images)

                # Compute task loss (e.g., MSE or cross-entropy)
                total_loss = dual_task_loss(student_output, teacher_output, distillation_loss, labels, beta=distillation_weight)

                mlflow.log_metric('train_distillation_loss', distillation_loss.item(), step=epoch+1)


            # Backpropagation and optimization
            total_loss.backward()
            # Log gradient norm for analysis
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item()

            all_outputs.append(student_output.cpu().detach().numpy() if role == 'student' else teacher_output.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        train_loss = running_loss / len(train_loader)
        train_r2_value = r2_score(all_labels, all_outputs, multioutput='uniform_average')  # 통합 R2 score
        train_accuracies = calculate_accuracy(all_labels, all_outputs, [1.0])
        average_train_accuracies = np.mean(train_accuracies)

        train_losses.append(train_loss) # epoch마다 train loss 저장
        train_r2.append(train_r2_value) # epoch마다 r2 score 저장


        mlflow.log_metric('train_loss', train_loss, step=epoch+1) # mlflow에 train loss 기록
        mlflow.log_metric('train_R2', train_r2_value, step=epoch+1) # mlflow에 train R2 기록
        mlflow.log_metric('train_gradient_norm', grad_norm, step=epoch+1)

        print(f"\nEpoch {epoch+1} Summary:")
        if fold:
            print(f"Fold: {fold}/{n_folds}")
        print(f"Train Loss: {train_loss:.3f}")
        print(f"Train R2: {train_r2_value:.3f}")
        print(f"Train Accuracies (±1.0): {train_accuracies}")
        print(f"Average Train Accuracy (±1.0): {average_train_accuracies:.3f}")


        ###### Validation ######
        model.eval()
        val_loss = 0.0
        all_val_outputs = []
        all_val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                
                if role == 'teacher':
                    teacher_output, teacher_features = model(images)
                    loss = loss_func(teacher_output, labels)
                elif role == 'student':
                    student_output, teacher_output, distillation_loss = model(images)  # Use both outputs
                    loss = loss_func(student_output, labels)

                    mlflow.log_metric('val_distillation_loss', distillation_loss.item(), step=epoch+1)


                val_loss += loss.item()

                all_val_outputs.append(student_output.cpu().detach().numpy() if role == 'student' else teacher_output.cpu().detach().numpy())
                all_val_labels.append(labels.cpu().detach().numpy())

            scheduler.step(val_loss)

            val_loss /= len(val_loader)
            
            all_val_outputs = np.concatenate(all_val_outputs, axis=0)
            all_val_labels = np.concatenate(all_val_labels, axis=0)
            val_r2_value = r2_score(all_val_labels, all_val_outputs, multioutput='uniform_average')
            val_accuracies_05 = calculate_accuracy(all_val_labels, all_val_outputs, [0.5])
            val_accuracies_10 = calculate_accuracy(all_val_labels, all_val_outputs, [1.0])
            val_accuracies_20 = calculate_accuracy(all_val_labels, all_val_outputs, [2.0])
            average_val_accuracies_05 = np.mean(val_accuracies_05)
            average_val_accuracies_10 = np.mean(val_accuracies_10)
            average_val_accuracies_20 = np.mean(val_accuracies_20)

            mlflow.log_metric('val_loss', val_loss, step=epoch + 1)
            mlflow.log_metric('val_R2', val_r2_value, step=epoch + 1)
            mlflow.log_metric('average_val_acc_05', average_val_accuracies_05, step=epoch + 1)
            mlflow.log_metric('average_val_acc_10', average_val_accuracies_10, step=epoch + 1)
            mlflow.log_metric('average_val_acc_20', average_val_accuracies_20, step=epoch + 1)

            for i, (acc_05, acc_10, acc_20) in enumerate(zip(val_accuracies_05, val_accuracies_10, val_accuracies_20)):
                label_name = label_names[i]
                mlflow.log_metric(f'val_acc_05_{label_name}', acc_05, step=epoch + 1)
                mlflow.log_metric(f'val_acc_10_{label_name}', acc_10, step=epoch + 1)
                mlflow.log_metric(f'val_acc_20_{label_name}', acc_20, step=epoch + 1)

            # 가장 낮은 val_loss를 가진 모델 저장
            if val_loss < best_val_loss and save_model:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(model, "best_model")
                print("model saved")

            val_losses.append(val_loss)
            val_r2.append(val_r2_value)
            val_acc.append((val_accuracies_05, val_accuracies_10, val_accuracies_20))
            average_val_acc.append((average_val_accuracies_05, average_val_accuracies_10, average_val_accuracies_20))

            print(f"\nValidation Results:")
            if fold:
                print(f"Fold: {fold}/{n_folds}")
            print(f"Epoch: {epoch+1}/{num_epochs}")
            print(f"Validation Loss: {val_loss:.3f}")
            print(f"Validation R2: {val_r2_value:.3f}")
            print(f"Validation Accuracies (±0.5): {val_accuracies_05}")
            print(f"Validation Accuracies (±1.0): {val_accuracies_10}")
            print(f"Validation Accuracies (±2.0): {val_accuracies_20}")
            print(f"Average Validation Accuracy (±0.5): {average_val_accuracies_05:.3f}")
            print(f"Average Validation Accuracy (±1.0): {average_val_accuracies_10:.3f}")
            print(f"Average Validation Accuracy (±2.0): {average_val_accuracies_20:.3f}")

    # Save the teacher model after training, if role is teacher
    if role == 'teacher':
        teacher_model_path = f'./saved_models/{tuned_teacher_model_name}.pth'
        
        # After training, save the teacher model
        torch.save(model.state_dict(), teacher_model_path)
        print(f"Teacher model saved at {teacher_model_path}")
        
        # Log the model to MLflow
        mlflow.pytorch.log_model(model, artifact_path="teacher_model")
        
    return [train_losses, train_r2, val_losses, val_r2, val_acc, average_val_acc]



# 정확도 계산
def calculate_accuracy(labels, outputs, tolerances):
    accuracies = []
    for tol in tolerances:
        acc_per_tol = []
        for i in range(labels.shape[1]):
            correct = ((labels[:, i] - tol) <= outputs[:, i]) & (outputs[:, i] <= (labels[:, i] + tol))
            accuracy = correct.sum() / len(labels)
            acc_per_tol.append(accuracy)
        accuracies.append(acc_per_tol)
    return accuracies[0] if len(tolerances) == 1 else accuracies

