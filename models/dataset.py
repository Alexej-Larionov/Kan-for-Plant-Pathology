import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

disease_columns = ["healthy", "pear_slug", "leaf_spot", "curl"]
severity_columns = ["severity_0", "severity_1", "severity_2", "severity_3", "severity_4"]

def stratified_split(df, test_size=0.2, random_state=42):
    X = df["file_name"].values
    y = df[disease_columns + severity_columns].values  

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(msss.split(X, y))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    return df_train, df_test


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, root_dir=None, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["image_path"]

        if self.root_dir is not None and not os.path.isabs(img_path):
            img_path = os.path.join(self.root_dir, img_path)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        diseases = torch.tensor(row[disease_columns].values.astype("float32"))

        severity = torch.tensor(np.argmax(row[severity_columns].values), dtype=torch.long)

        return image, diseases, severity



