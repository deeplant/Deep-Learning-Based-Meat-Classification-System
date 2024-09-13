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
    original_grade_count = len(df[df['grade'] == grade])
    flipped_rows = []
    
    for _, row in df[df['grade'] == grade].iterrows():
        flipped_row = row.copy()
        flipped_row['is_flipped'] = True
        flipped_rows.append(flipped_row)
    
    df_flipped = pd.DataFrame(flipped_rows)
    df = pd.concat([df, df_flipped], ignore_index=True)
    
    new_grade_count = len(df[df['grade'] == grade])
    
    print(f"Added flipped images for {grade}. Original count: {original_grade_count}, New total for {grade}: {new_grade_count}")
    
    return df