from SIDD import *
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import cv2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from SIDD.config.configuration import ConfigurationManager
from SIDD.entity.config_entity import DataPreparationConfig



i_transform = A.Compose([
    A.Resize(256, 1600),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])


m_transform = A.Compose([
    A.Resize(256, 1600),
    A.HorizontalFlip(p=0.5),

    ToTensorV2()
])


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config
    
    def get_dataloaders(self):

        df = pd.read_csv(self.config.train_csv_path)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        train_dataset = ImageDataset(
            train_df,
            self.config.train_images_path,
            self.config.height,
            self.config.width,
            self.config.num_classes,
            i_transform,
            m_transform
        )

        test_dataset = ImageDataset(
            test_df,
            self.config.train_images_path,
            self.config.height,
            self.config.width,
            self.config.num_classes,
            i_transform,
            m_transform
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=self.config.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=self.config.num_workers, pin_memory=True)

        return train_loader, test_loader
    
    
class ImageDataset(Dataset):

    def get_image(self, image_id):
        path = Path(self.image_path) / image_id
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def create_empty_mask(self):
        return np.zeros((self.height, self.width, self.num_classes), dtype=np.uint8)
    
    def rle_decode(self, mask_rle, shape):
        s = mask_rle.split()

        starts = np.asarray(s[0::2], dtype=int) - 1
        lengths = np.asarray(s[1::2], dtype=int)

        ends = starts + lengths

        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        for start, end in zip(starts, ends):
            img[start:end] = 1

        return img.reshape(shape, order='F')
    
    def create_mask(self, image_id):

        if image_id not in self.grouped.groups:
            return self.create_empty_mask()
        
        mask = self.create_empty_mask()
        group = self.grouped.get_group(image_id)

        for _, row in group.iterrows():
            class_id = row["ClassId"]
            rle = row["EncodedPixels"]

            if pd.isna(rle):
                continue

            single_mask = self.rle_decode(rle, (self.height, self.width))

            mask[:, :, class_id - 1] = single_mask

        return mask
    

    def __init__(self, df, image_path, height, width, num_classes, image_transform=None, mask_transform=None):
        self.df = df
        self.image_ids = sorted(self.df["ImageId"].unique())
        self.image_path = image_path
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.grouped = self.df.groupby("ImageId")
        self.height = height   
        self.width = width     
        self.num_classes = num_classes  

    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = self.get_image(image_id)
        mask = self.create_mask(image_id)
        if self.image_transform is not None:
            augmented = self.image_transform(image=image, mask=mask)
            image = augmented["image"].float()
            mask = augmented["mask"].float()
        
        return image, mask
    

try:
    config = ConfigurationManager()
    data_prep_config = config.get_data_preparation_config()
    data_prep = DataPreparation(config=data_prep_config)
    (train_loader, test_loader) = data_prep.get_dataloaders()
    logger.info("Train and Test DataLoader has been created")
except Exception as e:
    raise e