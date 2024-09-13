# train, val 1 epoch를 진행하는 함수
# train 모드에서는 train과 validation 진행
# evaluate 모드에서는 validation만 진행
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import r2_score
import mlflow

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
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_outputs.append(outputs.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        train_loss = running_loss / len(train_loader)
        train_r2_value = r2_score(all_labels, all_outputs, multioutput='uniform_average')  # 통합 R2 score
        train_accuracies = calculate_accuracy(all_labels, all_outputs)
        average_train_accuracies = np.mean(train_accuracies)

        train_losses.append(train_loss) # epoch마다 train loss 저장
        train_r2.append(train_r2_value) # epoch마다 r2 score 저장

        mlflow.log_metric('train_loss', train_loss, step=epoch+1) # mlflow에 train loss 기록
        mlflow.log_metric('train_R2', train_r2_value, step=epoch+1) # mlflow에 train R2 기록

        print(f"\nEpoch {epoch+1} Summary:")
        if fold:
            print(f"Fold: {fold}/{n_folds}")
        print(f"Train Loss: {running_loss / len(train_loader):.3f}")
        print(f"Train R2: {train_r2_value:.3f}")
        print(f"Train Accuracies: {train_accuracies}")
        print(f"Average Train Accuracy: {average_train_accuracies:.3f}")


        ###### Validation ######
        model.eval()
        val_loss = 0.0
        all_val_outputs = []
        all_val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                loss = loss_func(outputs, labels)

                val_loss += loss.item()
                all_val_outputs.append(outputs.cpu().detach().numpy())
                all_val_labels.append(labels.cpu().detach().numpy())

            scheduler.step(val_loss)

            val_loss /= len(val_loader)
            all_val_outputs = np.concatenate(all_val_outputs, axis=0)
            all_val_labels = np.concatenate(all_val_labels, axis=0)
            val_r2_value = r2_score(all_val_labels, all_val_outputs, multioutput='uniform_average')
            val_accuracies = calculate_accuracy(all_val_labels, all_val_outputs)
            average_val_accuracies = np.mean(val_accuracies)

            mlflow.log_metric('val_loss', val_loss, step=epoch + 1)
            mlflow.log_metric('val_R2', val_r2_value, step=epoch + 1)
            mlflow.log_metric('average_val_acc', average_val_accuracies, step=epoch + 1)

            for i, acc in enumerate(val_accuracies): # 각 라벨마다 정확도 출력 
                label_name = label_names[i]
                mlflow.log_metric(f'val_acc_{label_name}', acc, step=epoch + 1)

            # 가장 낮은 val_loss를 가진 모델 저장
            if val_loss < best_val_loss and save_model:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(model, "best_model")

            val_losses.append(val_loss)
            val_r2.append(val_r2_value)
            val_acc.append(val_accuracies)
            average_val_acc.append(average_val_accuracies)

            print(f"\nValidation Results:")
            if fold:
                print(f"Fold: {fold}/{n_folds}")
            print(f"Epoch: {epoch+1}/{num_epochs}")
            print(f"Validation Loss: {val_loss:.3f}")
            print(f"Validation R2: {val_r2_value:.3f}")
            print(f"Validation Accuracies: {val_accuracies}")
            print(f"Average Validation Accuracy: {average_val_accuracies:.3f}")

    return [train_losses, train_r2, val_losses, val_r2, val_acc, average_val_acc]



# 정확도 계산
def calculate_accuracy(labels, outputs):
    accuracies = []
    for i in range(labels.shape[1]):
        correct = ((labels[:, i] - 1) <= outputs[:, i]) & (outputs[:, i] <= (labels[:, i] + 1))
        accuracy = correct.sum() / len(labels)
        accuracies.append(accuracy)
    return accuracies