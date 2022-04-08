import torch
import os
import cv2
import numpy as np
import json

from torch.utils.data import Dataset

class FIATClassificationDataset(Dataset):
    def __init__(self, labelSourcePath, label_height, label_width, isColor=True, isNorm=False):
        super(FIATClassificationDataset, self).__init__()
        self.labelSourcePath = labelSourcePath
        self.labelCount = 0
        self.label_height = label_height
        self.label_width = label_width
        self.isColor = isColor
        self.isNorm = isNorm
        self.labelJsonArray = []

        if labelSourcePath.endswith('//') == False :
            self.labelSourcePath += "//"


        label_info_file = self.labelSourcePath + "___target_info.json"
        self.labelNames = []
        with open(label_info_file, encoding='utf-8-sig') as f:
            json_objet = json.load(f)
            if len(json_objet) == 0 :
                raise Exception('Label is not exists')
            for label in json_objet:
                print(label)
                self.labelNames.append(label['Name'])


        filelist = sorted(os.listdir(self.labelSourcePath))
        for labelName in filelist:
            if labelName == '.DS_Store': continue
            if labelName == "___target_info.json":continue
            if labelName.endswith('.json') == False:continue
            self.labelJsonArray.append(self.labelSourcePath + labelName)

        self.labelCount = len(self.labelJsonArray)


  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.labelJsonArray)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # shuffle(self.fullPaths)
        json_info = self.labelJsonArray[idx]

        with open(json_info, encoding='utf-8-sig') as f:
            json_objet = json.load(f)


            ##### input image load
            imageFile = self.labelSourcePath + json_objet['FileName']

            isOpencvColor = cv2.IMREAD_GRAYSCALE
            if self.isColor == True:
                isOpencvColor =cv2.IMREAD_COLOR

            image = cv2.imread(imageFile, flags=isOpencvColor)
            image = cv2.resize(image, dsize=(self.label_width, self.label_height), interpolation=cv2.INTER_AREA)
            image = image.transpose(2, 0, 1)  # (H, W, CH) -> (CH, H, W)
            image = np.ascontiguousarray(image)
            x = torch.tensor(image, dtype=torch.float32)


            torch_output = torch.zeros([len(self.labelNames)], dtype=torch.float32)
            for index in range(len(self.labelNames)):
                for label in json_objet['ClassCollection']:
                    if self.labelNames[index] == label["Name"]:
                        torch_output[index] = 1

            return x, torch_output