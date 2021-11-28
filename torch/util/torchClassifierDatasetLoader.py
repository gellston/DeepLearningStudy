import torch
import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class torchClassifierDatasetLoader(Dataset):
    def __init__(self, root, image_height, image_width):
        super(torchClassifierDatasetLoader, self).__init__()
        self.root = root
        self.labelPaths = []
        self.fullPaths = []
        self.labelCount = 0
        self.image_height = image_height
        self.image_width = image_width

        filelist = sorted(os.listdir(self.root))
        for labelName in filelist:
            if labelName == '.DS_Store': continue
            temp = self.root + '/' + labelName
            if os.path.isdir(temp):
                self.labelPaths.append([temp, int(labelName.split('_')[0])])
                self.labelCount = self.labelCount + 1

        for index in range(self.labelCount):
            item = self.labelPaths[index]
            path = item[0]
            list = os.listdir(item[0])
            for name in list:
                if name == '.DS_Store': continue
                self.fullPaths.append([path + '/' + name, item[1]])

  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.fullPaths)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # shuffle(self.fullPaths)
        label_item = self.fullPaths[idx]

        image_path = label_item[0]
        y = label_item[1]
        y = torch.tensor(int(y))

        # print(path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
        x = torch.FloatTensor(cv2.resize(image, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_AREA))
        x = x.permute([2, 0, 1]).float()

        return x, y