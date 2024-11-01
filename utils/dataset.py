# 이미지, 라벨 매칭된 데이터셋 반환하는 코드

from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from utils.transform import create_transform
import numpy as np


class MeatDataset(Dataset):
    def __init__(self, dataframe, config, is_train=True):
        self.dataframe = dataframe
        self.config = config
        self.is_train = is_train
        self.transforms = self._create_transforms()

    def __len__(self):
        return len(self.dataframe)
    
    def _create_transforms(self):
        transforms = []
        for dataset_config in self.config['datasets']:
            if self.is_train:
                transform_config = dataset_config['train_transform']
            else:
                transform_config = dataset_config['val_transform']
            transforms.append(create_transform(transform_config))
        return transforms
    
    def load_image(self, img_path):
        image = Image.open(img_path)
        if image.mode == 'RGBA':
            return image.convert('RGB')
        return image

    def process_image_group(self, row, input_columns, transform):
        images = []
        for col in input_columns:
            img_name = row[col]
            image = self.load_image(img_name)
            
            if row.get('is_flipped', False):
                image = TF.vflip(image)
            
            images.append(np.array(image))
        
        # 이미지들을 numpy 배열로 결합 (채널 방향으로)
        combined_image = np.concatenate(images, axis=2)
        combined_image = Image.fromarray(combined_image)
        
        if transform:
            combined_image = transform(combined_image)
        
        return combined_image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.dataframe.iloc[idx]
        
        processed_images = []
        for dataset_idx, dataset_config in enumerate(self.config['datasets']):
            input_columns = dataset_config['input_columns']
            transform = self.transforms[dataset_idx] if self.transforms else None
            
            processed_image = self.process_image_group(row, input_columns, transform)
            processed_images.append(processed_image)
        
        # 다른 input_columns 그룹의 이미지들을 채널 차원으로 연결
        final_image = torch.cat(processed_images, dim=0)
        
        label_columns = self.config['output_columns']
        labels = [row.get(col, float('nan')) for col in label_columns]
        labels = torch.tensor(labels, dtype=torch.float)
        
        return final_image, labels
