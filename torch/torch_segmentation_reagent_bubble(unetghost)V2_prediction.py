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
from model.UNetGhostV2 import UNetGhostV2

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



datasets = TorchSegmentationDatasetLoaderV2(root_path="E://용액 검사 데이터셋//20230630_세그먼테이션 버블 대통합//",
                                            image_height=400,
                                            image_width=144,
                                            classNum=1,
                                            skipClass=[1],
                                            isColor=False,
                                            isNorm=False,
                                            side_offset=0,
                                            use_augmentation=True)

data_loader = DataLoader(datasets, batch_size=1, shuffle=True)


UNetGhostV2 = UNetGhostV2(activation=torch.nn.Hardswish,
                          use_activation=True,
                          expansion_rate=1,
                          class_num=1).to(device)
UNetGhostV2Weight = torch.jit.load("C://Github//DeepLearningStudy//trained_model//UNetGhostV2(Reagent_UNetGhostV2)_top.pt")
UNetGhostV2.load_state_dict(UNetGhostV2Weight.state_dict())





for X, Y in data_loader:
    gpu_X = X.to(device)
    gpu_Y = Y.to(device)

    ##acc calculation
    UNetGhostV2.eval()
    prediction, _, _ = UNetGhostV2(gpu_X)

    original_image = X[0].permute(1, 2, 0).numpy().astype('uint8')
    original_image = original_image.squeeze(2)

    bubble = Y[0].permute(1, 2, 0).detach().cpu().numpy()
    bubble = np.where(bubble > 0.5, 255, 0).astype('uint8')
    bubble = bubble[:, :, 0]

    #residue = Y[0].permute(1, 2, 0).detach().cpu().numpy()
    #residue = np.where(residue > 0.5, 255, 0).astype('uint8')
    #residue = residue[:, :, 1]

    prediction_bubble = prediction[0].permute(1, 2, 0).detach().cpu().numpy()
    prediction_bubble = np.where(prediction_bubble > 0.5, 255, 0).astype('uint8')
    prediction_bubble = prediction_bubble[:, :, 0]

    #prediction_residue = prediction[0].permute(1, 2, 0).detach().cpu().numpy()
    #prediction_residue = np.where(prediction_residue > 0.5, 255, 0).astype('uint8')
    #prediction_residue = prediction_residue[:, :, 1]

    cv2.namedWindow("original", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('original', 144, 400)
    cv2.imshow('original', original_image)

    cv2.namedWindow("bubble", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('bubble', 144, 400)
    cv2.imshow('bubble', bubble)

    #cv2.namedWindow("residue", cv2.WINDOW_FREERATIO)
    #cv2.resizeWindow('residue', 512, 128)
    #cv2.imshow('residue', residue)

    cv2.namedWindow("prediction_bubble", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('prediction_bubble', 144, 400)
    cv2.imshow('prediction_bubble', prediction_bubble)

    #cv2.namedWindow("prediction_residue", cv2.WINDOW_FREERATIO)
    #cv2.resizeWindow('prediction_residue', 512, 128)
    #cv2.imshow('prediction_residue', prediction_residue)

    cv2.waitKey(300)



