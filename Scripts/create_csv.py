"""
csv 파일을 생성하는 코드
학습할 때 이미지 경로를 직접 설정하지 않고, csv 파일에서 경로를 불러옴
base_directory: 등급 폴더들이 있는 폴더 경로
label_directory: 등급별 라벨 엑셀 파일들이 있는 폴더 경로

csv 형식

grade, Marbling, Color, Texture, Surface_Moisture, Total, image_path

ex) 등심1++,8.0,6.0,8.0,4.0,8.0,./meat_dataset/등심1++/등심1++_00353.jpg


경로 설정 방법

이미지 경로 설정 방법:
 - base_directory: 등급 폴더가 있는 폴더의 경로 입력
 - image_directories: 각 등급에 맞는 base_directory 안의 등급 폴더 경로 입력
   ex) image_directories = {'등급': os.path.join(base_directory, '등급 파일 경로')}

라벨 경로 설정 방법:
 - label_directory: 엑셀 라벨 파일들이 있는 폴더 경로
 - excel_files: 각 등급에 맞는 라벨 엑셀 파일이름 입력


이미지 이름 형식: 등심1++_00001.jpg, 등심2_01234.jpg (숫자는 6자리로 패딩)

csv에 저장되는 경로 예시: ./dataset/meat_dataset/등심1++_00738.jpg
"""

import os
import pandas as pd

base_directory = "./meat_dataset/"
label_directory = "./meat_dataset/labels"
output_file = "default.csv"

# 이미지 파일 경로 설정
image_directories = {
    '등심1++': os.path.join(base_directory, '등심1++'),
    '등심1+': os.path.join(base_directory, '등심1+'),
    '등심1': os.path.join(base_directory, '등심1'),
    '등심2': os.path.join(base_directory, '등심2'),
    '등심3': os.path.join(base_directory, '등심3')
}
# 엑셀 파일 경로 설정
excel_files = {
    '등심1++': os.path.join(label_directory, 'label_1++.xlsx'),
    '등심1+': os.path.join(label_directory, 'label_1+.xlsx'),
    '등심1': os.path.join(label_directory, 'label_1.xlsx'),
    '등심2': os.path.join(label_directory, 'label_2.xlsx'),
    '등심3': os.path.join(label_directory, 'label_3.xlsx')
}


def find_header_row(file_path):
    required_columns = ['No', '등급', 'Marbling(마블링정도)', 'Color(색깔)', 'Texture(조직감)', 'Surface Moisture(표면육즙)', 'Total(기호도)']
    
    # 엑셀 파일을 열고 각 행을 확인
    for i in range(20):  # 처음 20행만 확인 (필요에 따라 조정 가능)
        df = pd.read_excel(file_path, header=i, nrows=1)
        if all(col in df.columns for col in required_columns):
            return i
    
    raise ValueError(f"Required columns not found in the first 20 rows of {file_path}")

def process_data(base_directory, image_directories, excel_files, output_file):
    dataframes = []
    for file_path in excel_files:
        try:
            header_row = find_header_row(file_path)
            df = pd.read_excel(file_path, header=header_row)
            
            df = df.dropna(subset=['No', '등급'])
            df = df.dropna(subset=['Marbling(마블링정도)', 'Color(색깔)', 'Texture(조직감)', 'Surface Moisture(표면육즙)', 'Total(기호도)'])
            
            columns_to_check = ['Marbling(마블링정도)', 'Color(색깔)', 'Texture(조직감)', 'Surface Moisture(표면육즙)', 'Total(기호도)']
            for column in columns_to_check:
                df = df[pd.to_numeric(df[column], errors='coerce').notnull()]
                df[column] = df[column].astype(float) * 2
            
            # '등급' 열을 기준으로 이미지 경로 설정
            df['image_path'] = df.apply(lambda row: os.path.join(image_directories[row['등급']], f"{row['등급']}_{int(row['No']):05d}.jpg"), axis=1)
            
            df = df[df['image_path'].apply(os.path.exists)]
            
            # 유효한 파일 경로가 몇 개인지 로그 출력
            print(f"Filtered valid image paths for {file_path}: {len(df)} / {len(df)}")
            
            dataframes.append(df)

        except ValueError as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    # 모든 데이터프레임 병합
    all_data = pd.concat(dataframes, ignore_index=True)
    all_data.columns = all_data.columns.str.replace(r'\(.*\)', '', regex=True).str.strip()
    all_data.columns = all_data.columns.str.replace(' ', '_')
    
    # 필요한 열만 선택 (Total 포함)
    all_data = all_data[['등급', 'Marbling', 'Color', 'Texture', 'Surface_Moisture', 'Total', 'image_path']]
    
    # '등급' 열 이름을 'grade'로 변경
    all_data = all_data.rename(columns={'등급': 'grade'})
    
    # CSV 파일로 저장
    all_data.to_csv(os.path.join(base_directory, output_file), index=False)
    print(f"Data saved to {os.path.join(base_directory, output_file)}")

    return all_data

processed_data = process_data(base_directory, image_directories, excel_files, output_file)