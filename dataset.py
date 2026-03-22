import os 
import pandas as pd 
#import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Transformation de l'entrainement
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.486, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transformation de la validation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.486, 0.406], std=[0.229, 0.224, 0.225]),
])

class CoinDataset(Dataset):
    def __init__(self, image_paths, labels, label2idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label2idx[self.labels[idx]]
        return img, label
    
def build_dataloaders(data_dir, csv_path, batch_size=32, val_size=0.2, seed=42):
    df = pd.read_csv(csv_path)

    df = df[df['Id'].apply(lambda x: os.path.exists(os.path.join(data_dir, f"{x}.jpg")))]
    print(f"Images valides: {len(df)}")

    image_paths = [os.path.join(data_dir, f"{row['Id']}.jpg") 
                       for _, row in df.iterrows()]
    labels = df['Class'].tolist()

    classes = sorted(set(labels))
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    num_classes = len(classes)
    print(f"Classes trouvées: ({num_classes})")

    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=val_size, stratify=labels, random_state=seed)

    train_dataset = CoinDataset(X_train, y_train, label2idx, train_transform)
    val_dataset = CoinDataset(X_val, y_val, label2idx, val_transform)

    train_Loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_Loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_Loader, valid_Loader, num_classes, idx2label

    

                       


