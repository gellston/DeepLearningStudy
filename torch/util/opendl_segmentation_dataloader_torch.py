import os
import cv2 as cv2
import numpy as np
import torch

from torch.utils.data import Dataset







class opendl_segmentation_dataloader_torch(Dataset):
    def __init__(self, labelSourcePath, label_height, label_width):
        super(opendl_segmentation_dataloader_torch, self).__init__()

        self.input_images_path = labelSourcePath
        self.label_height = label_height
        self.label_width = label_width
        self.labelCount = 0
        self.input_images_paths = []
        self.dataset_count = 0
        self.currentIndex = 0

        filelist = sorted(os.listdir(self.input_images_path))
        for filename in filelist:
            if filename == '.DS_Store': continue
            if filename == '__LabelInfo.json': continue

            source_file_name = self.input_images_path + '/' + filename + '/source.jpg'
            label_file_name = self.input_images_path + '/' + filename + '/0.jpg'

            self.input_images_paths.append([source_file_name, label_file_name])
            self.dataset_count += 1


        self.labelCount = self.dataset_count


  # 총 데이터의 개수를 리턴
    def __len__(self):
        return self.labelCount

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        source_file_name = self.input_images_paths[idx][0]
        label_file_name = self.input_images_paths[idx][1]

        isOpencvColor = cv2.IMREAD_GRAYSCALE

        source = cv2.imread(source_file_name, flags=isOpencvColor)
        source = cv2.resize(source, dsize=(self.label_width, self.label_height), interpolation=cv2.INTER_AREA)
        source = source.reshape((self.label_height, self.label_width, 1))
        source = source.transpose(2, 0, 1)  # (H, W, CH) -> (CH, H, W)
        source = np.ascontiguousarray(source)
        x = torch.tensor(source, dtype=torch.float32)



        target = cv2.imread(label_file_name, flags=isOpencvColor)
        target = cv2.resize(target, dsize=(self.label_width, self.label_height), interpolation=cv2.INTER_AREA)
        target = target.reshape((self.label_height, self.label_width, 1))
        target = target.transpose(2, 0, 1)  # (H, W, CH) -> (CH, H, W)
        target = np.ascontiguousarray(target)

        _, target = cv2.threshold(target, 128, 255, cv2.THRESH_BINARY)
        target = target / 255

        y = torch.tensor(target, dtype=torch.float32)

        return x, y




