import os
import cv2 as cv2
import numpy as np
from enum import Enum
from random import shuffle

class pathtype(Enum):
    relative = 1
    absolute = 2

class dataloader:
    def __init__(self, root):
        self.root = root
        self.fullPaths = []
        self.labelNames = []
        self.labelPaths = []
        self.labelCount = 0

        filelist = sorted(os.listdir(self.root))
        for labelName in filelist:
            if labelName == '.DS_Store': continue
            temp = self.root + '/' + labelName
            if os.path.isdir(temp):
                self.labelPaths.append([temp, labelName.split('_')[1]])
                self.labelCount = self.labelCount + 1



    def load(self, shape1, shape2, dev, batch, is_color=True):

        images = []
        labels = []

        for index in range(batch):
            if index + self.currentIndex >= self.size:
                return (None, None)

            #shuffle(self.fullPaths)
            path = self.fullPaths[self.currentIndex + index]
            color_flag = cv2.IMREAD_COLOR

            if is_color == True:
                color_flag = cv2.IMREAD_COLOR
            else:
                color_flag = cv2.IMREAD_GRAYSCALE

            #print(path)
            image = cv2.imread(path, color_flag).astype(np.uint8)
            ##print('image shape = ' , image.shape)
            #image = cv2.resize(image, (shape1[0], shape1[1]))
            npImage = np.array(image)
            npImage = npImage / dev
            npImage = npImage.flatten().reshape(shape1)
            npImage = np.array(npImage, dtype=np.uint8)
            images.append(npImage)

            #label = [0] * self.labelCount
            for index2 in range(self.labelCount):
                if self.labelNames[index2][0] in path:
                    #label[self.labelNames[index2][1]] = 1
                    #npLabel = np.array(label).flatten().reshape(shape2)
                    labels.append(index2)

            if index + self.currentIndex >= self.size:
                break

        self.currentIndex += batch
        numpy_image = np.array(images)
        numpy_label = np.array(labels)
        numpy_label = np.array(numpy_label).flatten().reshape(shape2)

        # print('current index =', self.currentIndex , '\n')
        return (numpy_image, numpy_label)

    def clear(self):
        self.currentIndex = 0

    def label_count(self):
        return self.labelCount

    def sample_count(self):
        return self.size

    def shuffle(self):
        shuffle(self.fullPaths)
