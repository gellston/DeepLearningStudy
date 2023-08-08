import torch
import os
import cv2
import numpy as np
import random

from torch.utils.data import Dataset


class TorchSegmentationDatasetLoaderV2(Dataset):
    def __init__(self,
                 root_path,
                 classNum,
                 skipClass,
                 image_height,
                 image_width,
                 isColor,
                 isNorm=False,
                 side_offset=65,
                 use_augmentation=False):

        super(TorchSegmentationDatasetLoaderV2, self).__init__()
        self.root_path = root_path
        self.classNum = classNum
        self.skipClass = skipClass
        self.image_name = []
        self.labelCount = 0
        self.image_height = image_height
        self.image_width = image_width
        self.isColor = isColor
        self.isNorm = isNorm
        self.side_offset = side_offset
        self.use_augmentation = use_augmentation

        filelist = sorted(os.listdir(self.root_path))

        for fileName in filelist:
            if fileName == '.DS_Store': continue
            if not fileName.endswith('.jpg'):continue
            fileNameToken = fileName.split('_')

            if fileNameToken[1] != 'original.jpg':
                continue

            fileNameStore = []
            fileNameStore.append(fileName)
            for index in range(self.classNum):
                fileNameStore.append(fileNameToken[0] + "_" + str(index) + ".jpg")
            self.image_name.append(fileNameStore)



  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.image_name)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # shuffle(self.fullPaths)
        original_image_name = self.image_name[idx]

        original_image = self.root_path + original_image_name[0]

        color_flag = cv2.IMREAD_COLOR

        if self.isColor is True:
            color_flag = cv2.IMREAD_COLOR
        else:
            color_flag = cv2.IMREAD_GRAYSCALE

        img_array = np.fromfile(original_image, np.uint8)  # 컴퓨터가 읽을수 있게 넘파이로 변환
        image = cv2.imdecode(img_array, color_flag).astype('uint8')


        rand_number = random.randrange(1, 3)



        if self.use_augmentation == True:
            if rand_number == 1:
                image = cv2.flip(image, 0)
            elif rand_number == 2:
                image = cv2.flip(image, 1)
            elif rand_number == 3:
                image = cv2.flip(image, 0)
                image = cv2.flip(image, 1)



        if self.isNorm is True:
            image = image / 255


        image = image[0:image.shape[0], self.side_offset:(image.shape[1] - self.side_offset)].copy()
        x = torch.FloatTensor(cv2.resize(image, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_AREA))
        x = x.unsqueeze(dim=2).float()
        x = x.permute([2, 0, 1])



        y_stack = []
        for index in range(self.classNum):

            skip_index_found = False
            for skip_index in self.skipClass:
                if index == skip_index:
                    skip_index_found = True

            if skip_index_found == True:
                continue



            label_image_name = self.root_path + self.image_name[idx][index + 1]
            img_array = np.fromfile(label_image_name, np.uint8)  # 컴퓨터가 읽을수 있게 넘파이로 변환
            label = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE).astype('uint8')

            if self.use_augmentation == True:
                if rand_number == 1:
                    label = cv2.flip(label, 0)
                elif rand_number == 2:
                    label = cv2.flip(label, 1)
                elif rand_number == 3:
                    label = cv2.flip(label, 0)
                    label = cv2.flip(label, 1)

            label = label[0:label.shape[0], self.side_offset:(label.shape[1] - self.side_offset)].copy()
            label = cv2.resize(label, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
            _, label = cv2.threshold(label, 128, 255, cv2.THRESH_BINARY)
            label = label / 255
            label = torch.FloatTensor(label)
            label = label.unsqueeze(dim=2).float()
            label = label.permute([2, 0, 1]).float()
            y_stack.append(label)

        y = torch.cat(y_stack)

        return x, y