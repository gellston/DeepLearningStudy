import torch
import os
import cv2
import numpy as np
import json

from torch.utils.data import Dataset


class FIATClassificationDataset(Dataset):
    def __init__(self, labelSourcePath, label_height, label_width, isColor, isNorm=False):
        super(FIATClassificationDataset, self).__init__()
        self.labelSourcePath = labelSourcePath
        self.labelCount = 0
        self.label_height = label_height
        self.label_width = label_width
        self.isColor = isColor
        self.isNorm = isNorm
        self.labelJsonArray = [];

        filelist = sorted(os.listdir(self.labelSourcePath))
        for labelName in filelist:
            if labelName == '.DS_Store': continue
            if labelName == "__LabelInfo.json": continue
            temp = self.labelSourcePath + '/' + labelName
            self.labelJsonArray.append(temp)
            self.labelCount = self.labelCount + 1

  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.labelJsonArray)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # shuffle(self.fullPaths)
        json_info = self.labelJsonArray[idx]
        startX = json_info.StartX
        startY = json_info.StartY
        endX = json_info.EndX
        endY = json_info.Endy
        imageWIdth = json_info.ImageWidth
        imageHeight = json_info.ImageHeight



        with open(json_info, encoding='utf-8-sig') as f:
            json_objet = json.load(f)


            print("check")



        return json_info, json_info