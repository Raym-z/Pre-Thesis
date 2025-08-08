import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

# ---- 3.1  Shared transforms ----------------------------------------------
IM_SIZE = 224  # 224×224 works for ResNet-50 & ViT
train_tf = T.Compose([
    T.RandomResizedCrop(IM_SIZE, scale=(0.8, 1.0)),      # augmentation
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.ToTensor(),                                        # to (C,H,W) float
    T.Normalize(mean=[0.485, 0.456, 0.406],              # ImageNet stats
                std=[0.229, 0.224, 0.225]),
])

val_tf = T.Compose([
    T.Resize(256),               # keep aspect ratio
    T.CenterCrop(IM_SIZE),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ---- 3.2  Dataset class ---------------------------------------------------
class StyleDetectDataset(Dataset):
    """Yields (tensor, label) from manifest rows."""
    def __init__(self, csv_file, split, transforms):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df.split == split].reset_index(drop=True)
        self.tf = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row.path).convert("RGB")  # always 3-channel
        label = int(row.label)
        return self.tf(img), label
