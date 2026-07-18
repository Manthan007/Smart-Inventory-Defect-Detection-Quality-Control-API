from SIDD import *
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import cv2
cv2.setNumThreads(0)
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
        self.df = pd.read_csv(self.config.train_csv_path)

        # 1. Gather all unique image IDs that contain defects from the CSV
        images_with_defects = self.df[self.df['EncodedPixels'].notna() & (self.df['EncodedPixels'] != '')]
        defect_images_list = images_with_defects['ImageId'].unique()
        
        # 2. Extract clean images directly from your input images directory
        all_physical_images = np.array([f.name for f in Path(self.config.train_images_path).glob("*.jpg")])
        clean_images_list = np.setdiff1d(all_physical_images, defect_images_list)

        # 3. Create the defect buckets from the CSV
        class_1_imgs = images_with_defects[images_with_defects['ClassId'] == 1]['ImageId'].unique()
        class_2_imgs = images_with_defects[images_with_defects['ClassId'] == 2]['ImageId'].unique()
        class_3_imgs = images_with_defects[images_with_defects['ClassId'] == 3]['ImageId'].unique()
        class_4_imgs = images_with_defects[images_with_defects['ClassId'] == 4]['ImageId'].unique()

        # 4. Stratified sampling with a guaranteed seed
        np.random.seed(42)
        sampled_c1 = np.random.choice(class_1_imgs, size=200, replace=False)
        sampled_c2 = np.random.choice(class_2_imgs, size=min(200, len(class_2_imgs)), replace=False)
        sampled_c3 = np.random.choice(class_3_imgs, size=200, replace=False)
        sampled_c4 = np.random.choice(class_4_imgs, size=200, replace=False)
        
        # This will now succeed because clean_images_list contains the thousands of flawless images
        sampled_clean = np.random.choice(clean_images_list, size=200, replace=False)

        # 5. Collect unique target IDs
        final_image_ids = np.concatenate([sampled_c1, sampled_c2, sampled_c3, sampled_c4, sampled_clean])
        final_image_ids = np.unique(final_image_ids)

        # Top up with clean images if minor overlaps drop the unique count slightly below 1000
        if len(final_image_ids) < 1000:
            extra_needed = 1000 - len(final_image_ids)
            remaining_clean = np.setdiff1d(clean_images_list, final_image_ids)
            extra_clean = np.random.choice(remaining_clean, size=extra_needed, replace=False)
            final_image_ids = np.concatenate([final_image_ids, extra_clean])

        # Overwrite self.image_ids with exactly 1,000 perfectly balanced IDs
        self.image_ids = sorted(list(final_image_ids))
        
        # Filter down the master dataframe to only keep entries for these 1,000 images
        self.df = self.df[self.df['ImageId'].isin(self.image_ids)].reset_index(drop=True)

        self.grouped = self.df.groupby("ImageId")
        self.image_annotations = {}

        for image_id, group in self.grouped:
            annotations = []
            for _, row in group.iterrows():
                if not pd.isna(row["EncodedPixels"]):
                    annotations.append((row["ClassId"], row["EncodedPixels"]))
            
            self.image_annotations[image_id] = annotations


        self.image_ids = sorted(self.df["ImageId"].unique())
    
    def get_dataloaders(self):

        train_ids, test_ids = train_test_split(
            self.image_ids,
            test_size=0.2,
            random_state=42
        )

        print("Total images:", len(self.image_ids))
        print("Train images:", len(train_ids))
        print("Test images:", len(test_ids))

        train_dataset = ImageDataset(
            image_annotations=self.image_annotations,
            image_ids=train_ids,
            image_path=self.config.train_images_path,
            height=self.config.height,
            width=self.config.width,
            num_classes=self.config.num_classes,
            image_transform=i_transform,
            mask_transform=m_transform
        )

        test_dataset = ImageDataset(
            image_annotations=self.image_annotations,
            image_ids=test_ids,
            image_path=self.config.train_images_path,
            height=self.config.height,
            width=self.config.width,
            num_classes=self.config.num_classes,
            image_transform=i_transform,
            mask_transform=m_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

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

        mask = np.zeros((self.height, self.width, self.num_classes), dtype=np.uint8)

        if image_id in self.image_annotations:
            for class_id, rle in self.image_annotations[image_id]:
                single_mask = self.rle_decode(rle, (self.height, self.width))
                mask[:, :, class_id-1] = single_mask
        return mask        

    def __init__(
        self,
        image_annotations,
        image_ids,
        image_path,
        height,
        width,
        num_classes,
        image_transform=None,
        mask_transform=None
    ):
        self.image_annotations = image_annotations

        self.image_ids = sorted(image_ids)

        self.image_path = image_path
        self.image_transform = image_transform
        self.mask_transform = mask_transform

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
            augmented = self.image_transform(
                image=image,
                mask=mask
            )

            image = augmented["image"].float()
            mask = augmented["mask"].float()

            mask = mask.permute(2, 0, 1)

        return image, mask
    

if __name__ == '__main__':
    try:
        config = ConfigurationManager()
        data_prep_config = config.get_data_preparation_config()
        data_prep = DataPreparation(config=data_prep_config)
        (train_loader, test_loader) = data_prep.get_dataloaders()
        logger.info("Train and Test DataLoader has been created")
    except Exception as e:
        raise e