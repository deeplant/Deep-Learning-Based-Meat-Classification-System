import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def split_data(csv, output_columns, cross_validation=0, flip_grade='등심3'):
    csv = add_flipped_images_to_dataset(csv, grade=flip_grade)

    if cross_validation == 0:
        n_splits = 5 
    else:
        n_splits = cross_validation

    fold_data = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for grade in csv['grade'].unique():
        grade_data = csv[csv['grade'] == grade].copy()
        
        scaler = StandardScaler()
        normalized_labels = scaler.fit_transform(grade_data[output_columns])
        
        grade_data['combined_score'] = np.mean(normalized_labels, axis=1)
        
        grouped = grade_data.groupby('No')
        group_scores = grouped['combined_score'].mean().sort_values()
        group_sizes = grouped.size()
        sorted_groups = list(group_scores.index)
        
        for no in sorted_groups:
            group_size = group_sizes[no]
            group_data = grade_data[grade_data['No'] == no]
            
            target_fold = min(range(n_splits), key=lambda i: fold_sizes[i])
            
            fold_data[target_fold].extend(group_data.to_dict('records'))
            fold_sizes[target_fold] += group_size

    fold_data = [pd.DataFrame(fold) for fold in fold_data]

    for fold in fold_data:
        fold.reset_index(drop=True, inplace=True)
        fold.drop(columns=['combined_score'], inplace=True, errors='ignore')

    return fold_data


def add_flipped_images_to_dataset(df, grade='등심3'):
    flipped_rows = []
    
    for _, row in df[df['grade'] == grade].iterrows():
        flipped_row = row.copy()
        flipped_row['is_flipped'] = True
        flipped_rows.append(flipped_row)
    
    df_flipped = pd.DataFrame(flipped_rows)
    df = pd.concat([df, df_flipped], ignore_index=True)
    
    print(f"Added flipped images for {grade}. Original count: {len(df) - len(flipped_rows)}, New total: {len(df)}")
    
    return df

    # if cross_validation == 0:
    #     # Train set (4개 폴드 결합)
    #     train_data = pd.concat(fold_data[:4], axis=0).reset_index(drop=True)
    #     # Validation set (1개 폴드)
    #     val_data = fold_data[4].reset_index(drop=True)
    #     return train_data, val_data
    # else:
    #     return fold_data