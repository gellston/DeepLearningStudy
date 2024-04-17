import torch
import os
import cv2
import numpy as np
import json

from torch.utils.data import Dataset

class MVTecAnomalyDataset(Dataset):
    def __init__(self, imagePath, image_height, image_width, isColor=True):
        super(MVTecAnomalyDataset, self).__init__()
        self.imagePath = imagePath
        self.labelCount = 0
        self.image_height = image_height
        self.image_width = image_width
        self.isColor = isColor
        self.imagePathArray = []

        if self.imagePath.endswith('//') == False :
            self.imagePath += "//"

        filelist = sorted(os.listdir(self.imagePath))
        for labelName in filelist:
            if labelName == '.DS_Store': continue
            if labelName.endswith('.png') == True:
                self.imagePathArray.append(self.imagePath + labelName)
            if labelName.endswith('.jpeg') == True:
                self.imagePathArray.append(self.imagePath + labelName)
            if labelName.endswith('.jpg') == True:
                self.imagePathArray.append(self.imagePath + labelName)
            if labelName.endswith('.bmp') == True:
                self.imagePathArray.append(self.imagePath + labelName)

        self.imageCount = len(self.imagePathArray)


  # 총 데이터의 개수를 리턴
    def __len__(self):
        return self.imageCount

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        imageFile = self.imagePathArray[idx]



        isOpencvColor = cv2.IMREAD_GRAYSCALE
        if self.isColor == True:
            isOpencvColor =cv2.IMREAD_COLOR

        image = cv2.imdecode(np.fromfile(imageFile, dtype=np.uint8), isOpencvColor)
        image = cv2.resize(image, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        if self.isColor == False:
            image = np.expand_dims(image, axis=2)
        image = image.transpose(2, 0, 1)  # (H, W, CH) -> (CH, H, W)
        image = np.ascontiguousarray(image)
        x = torch.tensor(image, dtype=torch.float32)
        return x


            