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
from util.losses import FocalLoss
from util.losses import DiceLoss
from util.losses import TverskyLoss
from util.losses import FocalTverskyLoss

from util.TorchSegmentationDatasetLoaderV2 import TorchSegmentationDatasetLoaderV2
from model.ParticleDetectorV1 import ParticleDetectorV1

gc.collect()
torch.cuda.set_per_process_memory_fraction(1.0)
torch.cuda.empty_cache()

USE_CUDA = torch.cuda.is_available()  # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu")  # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

training_epochs = 500
batch_size = 24

target_accuracy = 0.97
learning_rate = 0.003
final_learning_rate = learning_rate * batch_size / 64
classNum = 1

datasets = TorchSegmentationDatasetLoaderV2(root_path="D://프로젝트//시약검사//이미지//20230611 파티클 영상 후처리//",
                                            image_height=336,
                                            image_width=736,
                                            classNum=classNum,
                                            skipClass=[],
                                            isColor=False,
                                            isNorm=False)

data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

ParticleDetectorV1 = ParticleDetectorV1(class_num=classNum,
                              activation=torch.nn.LeakyReLU).to(device)
#BiSegNetWeight = torch.jit.load("C://Github//DeepLearningStudy//trained_model//BiSegNet(ReagentV4)_top.pt")
#BiSegNet.load_state_dict(BiSegNetWeight.state_dict())

print('==== model info ====')
summary(ParticleDetectorV1, (1, 736, 336))
ParticleDetectorV1.eval()
compiled_model = torch.jit.script(ParticleDetectorV1)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//ParticleDetectorV1_no_train.pt")
print('====================')

# loss_fn = nn.BCELoss().to(device)
# loss_fn = FocalLoss().to(device)
# loss_fn = DiceLoss().to(device)
loss_fn = TverskyLoss(alpha=0.3, beta=0.7).to(device)
#loss_fn = FocalTverskyLoss(alpha=0.45, beta=0.56, gamma=2)

optimizer = torch.optim.RAdam(ParticleDetectorV1.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('total_batch=', total_batch)

final_cost = 0
final_accuracy = 0

top_accuracy = 0

for epoch in range(training_epochs):

    sum_cost = 0
    sum_acc = 0

    for X, Y in data_loader:
        gpu_X = X.to(device)
        gpu_Y = Y.to(device)

        ##cost calculation
        ParticleDetectorV1.train()
        optimizer.zero_grad()
        result = ParticleDetectorV1(gpu_X)

        loss = loss_fn(result, gpu_Y)
        total_loss = loss
        total_loss.backward()
        optimizer.step()

        ##acc calculation
        ParticleDetectorV1.eval()
        prediction = ParticleDetectorV1(gpu_X)

        ground_numpy = np.asarray(tuple(t.cpu().detach().numpy() for t in gpu_Y))
        prediction_numpy = np.asarray(tuple(t.cpu().detach().numpy() for t in prediction))

        accuracy = IOU(ground_numpy, prediction_numpy)

        del gpu_X
        del gpu_Y
        torch.cuda.empty_cache()

        sum_cost += total_loss.item()
        sum_acc += accuracy

        gc.collect()

    ParticleDetectorV1.eval()
    compiled_model = torch.jit.script(ParticleDetectorV1)
    torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//ParticleDetectorV1_step.pt")

    average_cost = sum_cost / total_batch
    average_accuracy = sum_acc / total_batch

    print('=========================================')
    print('average cost=', average_cost)
    print('average accuracy=', average_accuracy)
    print('current epoch = ', epoch)
    print('=========================================')

    if top_accuracy <= average_accuracy:
        top_accuracy = average_accuracy
        ParticleDetectorV1.eval()
        compiled_model = torch.jit.script(ParticleDetectorV1)
        torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//ParticleDetectorV1_top.pt")

    if average_accuracy >= target_accuracy:
        ParticleDetectorV1.eval()
        compiled_model = torch.jit.script(ParticleDetectorV1)
        torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//ParticleDetectorV1_final.pt")
        break

    original_image = X[0].permute(1, 2, 0).numpy().astype('uint8')
    original_image = original_image.squeeze(2)

    bubble = Y[0].permute(1, 2, 0).detach().cpu().numpy()
    bubble = np.where(bubble > 0.5, 255, 0).astype('uint8')
    bubble = bubble[:, :, 0]

    prediction_cross_mark = prediction[0].permute(1, 2, 0).detach().cpu().numpy()
    prediction_cross_mark = np.where(prediction_cross_mark > 0.5, 255, 0).astype('uint8')
    prediction_cross_mark = prediction_cross_mark[:, :, 0]


    cv2.namedWindow("original", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('original', 736, 336)
    cv2.imshow('original', original_image)

    cv2.namedWindow("particle", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('particle', 736, 336)
    cv2.imshow('particle', bubble)


    cv2.namedWindow("prediction_particle", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('prediction_particle', 736, 336)
    cv2.imshow('prediction_particle', prediction_cross_mark)


    cv2.waitKey(100)

print('training finished')
print('top accuracy = ', top_accuracy)
