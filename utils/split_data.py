import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def split_data(csv, output_columns, cross_validation=0):

    if cross_validation == 0:
        n_splits = 5 
    else:
        n_splits = cross_validation

    fold_data = [[] for _ in range(n_splits)]

    for grade in csv['grade'].unique():
        grade_data = csv[csv['grade'] == grade].copy()
        
        scaler = StandardScaler()
        normalized_labels = scaler.fit_transform(grade_data[output_columns])
        
        grade_data['combined_score'] = np.mean(normalized_labels, axis=1)
        
        grade_data = grade_data.sort_values('combined_score')
        
        grade_data['group'] = pd.qcut(grade_data['combined_score'], q=n_splits, labels=False)
        
        for i, row in enumerate(grade_data.itertuples()):
            fold_index = i % n_splits
            fold_data[fold_index].append(row)

    fold_data = [pd.DataFrame(fold) for fold in fold_data]

    for fold in fold_data:
        fold.reset_index(drop=True, inplace=True)
        fold.drop(columns=['Index', 'group', 'combined_score'], inplace=True)

    return fold_data

    # if cross_validation == 0:
    #     # Train set (4개 폴드 결합)
    #     train_data = pd.concat(fold_data[:4], axis=0).reset_index(drop=True)
    #     # Validation set (1개 폴드)
    #     val_data = fold_data[4].reset_index(drop=True)
    #     return train_data, val_data
    # else:
    #     return fold_data