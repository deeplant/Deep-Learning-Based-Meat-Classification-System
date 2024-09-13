import numpy as np
import mlflow

def log(all_fold_results, cross_validation, label_names):
    num_folds = len(all_fold_results)
    num_epochs = len(all_fold_results[0][0])
    num_labels = len(all_fold_results[0][4][0])

    train_losses = np.array([fold_result[0] for fold_result in all_fold_results])          # (num_folds, num_epochs)
    train_r2 = np.array([fold_result[1] for fold_result in all_fold_results])              # (num_folds, num_epochs)
    val_losses = np.array([fold_result[2] for fold_result in all_fold_results])            # (num_folds, num_epochs)
    val_r2 = np.array([fold_result[3] for fold_result in all_fold_results])                # (num_folds, num_epochs)
    val_acc = np.array([fold_result[4] for fold_result in all_fold_results])               # (num_folds, num_epochs, num_labels)
    average_val_acc = np.array([fold_result[5] for fold_result in all_fold_results])       # (num_folds, num_epochs)

    # epoch별로 폴드들의 평균 값 계산
    average_train_losses = train_losses.mean(axis=0)           # (num_epochs,)
    average_train_r2 = train_r2.mean(axis=0)                   # (num_epochs,)
    average_val_losses = val_losses.mean(axis=0)               # (num_epochs,)
    average_val_r2 = val_r2.mean(axis=0)                       # (num_epochs,)
    average_val_acc_per_label = val_acc.mean(axis=0)           # (num_epochs, num_labels)
    average_average_val_acc = average_val_acc.mean(axis=0)     # (num_epochs,)

    if cross_validation:
        for epoch in range(num_epochs):
            mlflow.log_metric('average_train_loss', average_train_losses[epoch], step=epoch + 1)
            mlflow.log_metric('average_train_r2', average_train_r2[epoch], step=epoch + 1)
            mlflow.log_metric('average_val_loss', average_val_losses[epoch], step=epoch + 1)
            mlflow.log_metric('average_val_r2', average_val_r2[epoch], step=epoch + 1)
            mlflow.log_metric('average_average_val_acc', average_average_val_acc[epoch], step=epoch + 1)
            
            for i in range(num_labels):
                label_name = label_names[i]
                mlflow.log_metric(f'average_val_acc_{label_name}', average_val_acc_per_label[epoch, i], step=epoch + 1)

    min_val_loss_idx = np.argmin(val_losses)
    best_fold_idx, best_epoch_idx = np.unravel_index(min_val_loss_idx, val_losses.shape)

    best_val_loss = val_losses[best_fold_idx, best_epoch_idx]
    best_val_r2 = val_r2[best_fold_idx, best_epoch_idx]
    best_val_acc_per_label = val_acc[best_fold_idx, best_epoch_idx]
    best_average_val_acc = average_val_acc[best_fold_idx, best_epoch_idx]

    mlflow.log_metric('best_val_loss', best_val_loss)
    mlflow.log_metric('best_val_r2', best_val_r2)
    mlflow.log_metric('best_average_val_acc', best_average_val_acc)

    for i in range(num_labels):
        label_name = label_names[i]
        mlflow.log_metric(f'best_val_acc_{label_name}', best_val_acc_per_label[i])

    # 결과 출력
    print(f"\nBest Validation Results:")
    print(f"Fold: {best_fold_idx + 1}/{num_folds}")
    print(f"Epoch: {best_epoch_idx + 1}/{num_epochs}")
    print(f"Validation Loss: {best_val_loss:.3f}")
    print(f"Validation R2: {best_val_r2:.3f}")
    print(f"Validation Accuracies: {best_val_acc_per_label}")
    print(f"Average Validation Accuracy: {best_average_val_acc:.3f}")