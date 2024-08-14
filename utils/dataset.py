# 이미지, 라벨 매칭된 데이터셋 반환하는 코드

####### 예제 코드 #######
class MeatDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.dataframe.iloc[idx]
        
        img_name = row['image_path']
        image = Image.open(img_name).convert('RGB')
        
        if row.get('is_flipped', False):
            image = vflip(image)
        
        if self.transform:
            image = self.transform(image)
        
        label_columns = ['Marbling', 'Color', 'Texture', 'Surface_Moisture', 'Total']
        labels = [row.get(col, float('nan')) for col in label_columns]
        labels = torch.tensor(labels, dtype=torch.float)
        
        
        
        return image, labels