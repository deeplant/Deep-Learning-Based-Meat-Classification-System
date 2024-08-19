# transform 저장하는 코드

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class MultiChannelMeatDataset(Dataset):
    def __init__(self, dataframes, configs):
        self.datasets = [MeatDataset(df, config, self._create_transform(config['transform']))
                         for df, config in zip(dataframes, configs)]

    def _create_transform(self, transform_config):
        transforms = []
        if transform_config.get('RandomRotation'):
            transforms.append(TF.RandomRotation(transform_config['RandomRotation']))
        if transform_config.get('Resize'):
            transforms.append(TF.Resize(transform_config['Resize']))
        if transform_config.get('CenterCrop'):
            transforms.append(TF.CenterCrop(transform_config['CenterCrop']))
        if transform_config.get('RandomHorizontalFlip'):
            transforms.append(TF.RandomHorizontalFlip(transform_config['RandomHorizontalFlip']))
        if transform_config.get('ToTensor', False):
            transforms.append(TF.ToTensor())
        if transform_config.get('Normalize', False):
            transforms.append(TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return TF.Compose(transforms)

    def __len__(self):
        return min(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        images = []
        labels = []
        for dataset in self.datasets:
            image, label = dataset[idx]
            images.append(image)
            labels.append(label)
        
        # 이미지를 채널 방향으로 결합
        combined_image = torch.cat(images, dim=0)
        combined_labels = torch.stack(labels)
        
        return combined_image, combined_labels