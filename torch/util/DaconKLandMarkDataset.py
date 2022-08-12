import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import cv2
from torchvision import transforms


class DaconKLandMarkDataset(data.Dataset):
    def __init__(self,
                 data_root,
                 image_width,
                 image_height,
                 no_augmentation=False,
                 ops='train'):

        super(DaconKLandMarkDataset, self).__init__()

        self.data_root = data_root
        self.train_image_root = self.data_root + 'train//'
        self.test_image_root = self.data_root + 'test//'
        self.no_augmentation = no_augmentation

        self.image_width = image_width
        self.image_height = image_height


        if ops == 'train':
            self.target_csv_path = self.data_root + 'train.csv'
            self.target_image_path = self.train_image_root

        if ops == 'test':
            self.target_csv_path = self.data_root + 'test.csv'
            self.target_image_path = self.test_image_root
        self.ops = ops
        df = pd.read_csv(self.target_csv_path)
        # df.shape : (1314,16)

        self.row_count = df.shape[0]

        self.inp = df.iloc[:, 0:1].values
        self.outp = df.iloc[:, 1:2].values

        self.torchvision_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.RandomErasing(p=0.6, scale=(0.02, 0.05), ratio=(0.3, 1.3)),
            transforms.RandomErasing(p=0.6, scale=(0.02, 0.05), ratio=(0.3, 1.3)),
            transforms.RandomErasing(p=0.6, scale=(0.02, 0.05), ratio=(0.3, 1.3)),
            transforms.RandomErasing(p=0.6, scale=(0.02, 0.05), ratio=(0.3, 1.3)),
            transforms.RandomErasing(p=0.6, scale=(0.02, 0.05), ratio=(0.3, 1.3)),
            transforms.RandomErasing(p=0.6, scale=(0.02, 0.05), ratio=(0.3, 1.3)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            transforms.RandomResizedCrop(size=(960, 640), scale=(0.3, 2.5), interpolation=transforms.InterpolationMode.BICUBIC),

            transforms.Resize(size=(self.image_height, self.image_width)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


        self.torchvision_transform4test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(self.image_height, self.image_width)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.torchvision_transform4train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(self.image_height, self.image_width)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])




    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imagePath = self.inp[idx][0]



        imagePath = self.target_image_path + imagePath

        image = cv2.imread(imagePath, flags=cv2.IMREAD_COLOR)
        #image = cv2.resize(image, interpolation=cv2.INTER_LANCZOS4, dsize=(self.image_width, self.image_height))
        if self.ops == 'test':
            image = self.torchvision_transform4test(image)
            return image

        if self.no_augmentation == False:
            image = self.torchvision_transform(image)
        else:
            image = self.torchvision_transform4train(image)



        if self.ops == 'train':
            outputIndex = self.outp[idx][0]
            torch_output = torch.zeros([10], dtype=torch.float32)
            torch_output[outputIndex] = 1

            return image, torch_output

    def __len__(self):
        return self.row_count