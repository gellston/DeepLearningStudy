import gc
import torch
import torch.nn as nn
import random
import numpy as np
import cv2
import torch.nn.functional as F


from torchsummary import summary
from torch.utils.data import DataLoader
from util.helper import IOU
from util.losses import jaccard_loss

from util.TorchSegmentationDatasetLoaderV2 import TorchSegmentationDatasetLoaderV2
from model.BiSeNetMobileV2 import BiSeNetMobileV2

gc.collect()
torch.cuda.set_per_process_memory_fraction(1.0)
torch.cuda.empty_cache()


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)



datasets = TorchSegmentationDatasetLoaderV2(root_path="D://프로젝트//시약검사//이미지//20230601_3cc 고무_크롭_Segmentation_전처리//",
                                            image_height=416,
                                            image_width=160,
                                            classNum=2,
                                            skipClass=[0],
                                            isColor=False,
                                            isNorm=False)

data_loader = DataLoader(datasets, batch_size=1, shuffle=True)


BiSegNet = BiSeNetMobileV2(class_num=1,
                           activation=torch.nn.ReLU6).to(device)
BiSegNetWeight = torch.jit.load("C://Github//DeepLearningStudy//trained_model//BiSegNet(ReagentV1)_top.pt")
BiSegNet.load_state_dict(BiSegNetWeight.state_dict())





for X, Y in data_loader:
    gpu_X = X.to(device)
    gpu_Y = Y.to(device)

    ##acc calculation
    BiSegNet.eval()
    prediction, _, _ = BiSegNet(gpu_X)


    original_image = X[0].permute(1, 2, 0).numpy().astype('uint8')
    original_image = original_image.squeeze(2)


    prediction_image = prediction[0].permute(1, 2, 0).detach().cpu().numpy()
    prediction_image = np.where(prediction_image > 0.5, 255, 0).astype('uint8')
    prediction_image = prediction_image.squeeze(2)


    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original', 160, 416)
    cv2.imshow('original', original_image)

    cv2.namedWindow("prediction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('prediction', 160, 416)
    cv2.imshow('prediction', prediction_image)

    cv2.waitKey(300)



