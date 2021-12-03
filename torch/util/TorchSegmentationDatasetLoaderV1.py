import torch
import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class TorchSegmentationDatasetLoaderV1(Dataset):
    def __init__(self, original_image_path, label_image_path, image_height, image_width, isColor, isNorm=False):
        super(TorchSegmentationDatasetLoaderV1, self).__init__()
        self.original_image_path = original_image_path
        self.label_image_path = label_image_path
        self.origianlPaths = []
        self.labelPaths = []
        self.fullPaths = []
        self.labelCount = 0
        self.image_height = image_height
        self.image_width = image_width
        self.isColor = isColor
        self.isNorm = isNorm

        filelist = sorted(os.listdir(self.original_image_path))
        for labelName in filelist:
            if labelName == '.DS_Store': continue
            temp = self.original_image_path + '/' + labelName
            self.origianlPaths.append(temp)
            self.labelCount = self.labelCount + 1

        filelist = sorted(os.listdir(self.label_image_path))
        for labelName in filelist:
            if labelName == '.DS_Store': continue
            temp = self.label_image_path + '/' + labelName
            self.labelPaths.append(temp)

        for index in range(self.labelCount):
            self.fullPaths.append([self.origianlPaths[index], self.labelPaths[index]])

  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.fullPaths)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # shuffle(self.fullPaths)
        label_item = self.fullPaths[idx]

        original_image = label_item[0]
        label_image = label_item[1]


        color_flag = cv2.IMREAD_COLOR

        if self.isColor is True:
            color_flag = cv2.IMREAD_COLOR
        else:
            color_flag = cv2.IMREAD_GRAYSCALE

        image = cv2.imread(original_image, color_flag).astype('uint8')
        if self.isNorm is True:
            image = image / 255
        x = torch.FloatTensor(cv2.resize(image, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_AREA))
        x = x.permute([2, 0, 1])

        label = cv2.imread(label_image, cv2.IMREAD_GRAYSCALE).astype('uint8')
        _, label = cv2.threshold(label, 20, 255, cv2.THRESH_BINARY)
        label = label / 255

        y = torch.FloatTensor(cv2.resize(label, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_AREA))
        y = y.unsqueeze(dim=0)
        ##y = y.permute([2, 0, 1]).float()

        return x, y