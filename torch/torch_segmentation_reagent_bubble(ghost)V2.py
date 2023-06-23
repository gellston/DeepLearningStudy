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
from model.BiSegNetGhostV2 import BiSegNetGhostV2

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
batch_size = 15

target_accuracy = 0.99
learning_rate = 0.02
learning_rate_gamma = 0.5
scheduler_step = 10
classNum = 2

datasets = TorchSegmentationDatasetLoaderV2(root_path="D://프로젝트//시약검사//이미지//세그먼테이션 후처리 병합//",
                                            image_height=128,
                                            image_width=512,
                                            classNum=classNum,
                                            skipClass=[],
                                            isColor=False,
                                            isNorm=False,
                                            side_offset=65)

data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

BiSegNet = BiSegNetGhostV2(class_num=classNum,
                           activation=torch.nn.ReLU,
                           use_activation=True).to(device)
BiSegNetWeight = torch.jit.load("C://Github//DeepLearningStudy//trained_model//BiSegNet(Reagent_GhostV2)_top.pt")
BiSegNet.load_state_dict(BiSegNetWeight.state_dict())
print('==== model info ====')
summary(BiSegNet, (1, 512, 128))
BiSegNet.eval()
compiled_model = torch.jit.script(BiSegNet)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//BiSegNet(Reagent_GhostV2)_no_train.pt")
print('====================')

# loss_fn = nn.BCELoss().to(device)
# loss_fn = FocalLoss().to(device)
# loss_fn = DiceLoss().to(device)
# loss_fn = TverskyLoss(alpha=0.2, beta=0.8).to(device)
loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=2)

optimizer = torch.optim.RAdam(BiSegNet.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=learning_rate_gamma)

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
        BiSegNet.train()
        optimizer.zero_grad()
        result, result_spv1, result_spv2 = BiSegNet(gpu_X)

        loss1 = loss_fn(result, gpu_Y)
        loss2 = loss_fn(result_spv1, gpu_Y)
        loss3 = loss_fn(result_spv2, gpu_Y)
        total_loss = loss1 + loss2 + loss3
        total_loss.backward()
        optimizer.step()

        ##acc calculation
        BiSegNet.eval()
        prediction, _, _ = BiSegNet(gpu_X)

        ground_numpy = np.asarray(tuple(t.cpu().detach().numpy() for t in gpu_Y))
        prediction_numpy = np.asarray(tuple(t.cpu().detach().numpy() for t in prediction))

        accuracy = IOU(ground_numpy, prediction_numpy)

        del gpu_X
        del gpu_Y
        torch.cuda.empty_cache()

        sum_cost += total_loss.item()
        sum_acc += accuracy

        gc.collect()

    scheduler.step()
    print('current learning rate = ', scheduler.get_last_lr())

    BiSegNet.eval()
    compiled_model = torch.jit.script(BiSegNet)
    torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//BiSegNet(Reagent_GhostV2)_step.pt")

    average_cost = sum_cost / total_batch
    average_accuracy = sum_acc / total_batch

    print('=========================================')
    print('average cost=', average_cost)
    print('average accuracy=', average_accuracy)
    print('current epoch = ', epoch)
    print('=========================================')

    if top_accuracy <= average_accuracy:
        top_accuracy = average_accuracy
        BiSegNet.eval()
        compiled_model = torch.jit.script(BiSegNet)
        torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//BiSegNet(Reagent_GhostV2)_top.pt")

    if average_accuracy >= target_accuracy:
        BiSegNet.eval()
        compiled_model = torch.jit.script(BiSegNet)
        torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//BiSegNet(Reagent_GhostV2)_final.pt")
        break

    original_image = X[0].permute(1, 2, 0).numpy().astype('uint8')
    original_image = original_image.squeeze(2)

    bubble = Y[0].permute(1, 2, 0).detach().cpu().numpy()
    bubble = np.where(bubble > 0.5, 255, 0).astype('uint8')
    bubble = bubble[:, :, 0]

    residue = Y[0].permute(1, 2, 0).detach().cpu().numpy()
    residue = np.where(residue > 0.5, 255, 0).astype('uint8')
    residue = residue[:, :, 1]

    prediction_bubble = prediction[0].permute(1, 2, 0).detach().cpu().numpy()
    prediction_bubble = np.where(prediction_bubble > 0.5, 255, 0).astype('uint8')
    prediction_bubble = prediction_bubble[:, :, 0]

    prediction_residue = prediction[0].permute(1, 2, 0).detach().cpu().numpy()
    prediction_residue = np.where(prediction_residue > 0.5, 255, 0).astype('uint8')
    prediction_residue = prediction_residue[:, :, 1]

    cv2.namedWindow("original", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('original', 512, 128)
    cv2.imshow('original', original_image)

    cv2.namedWindow("bubble", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('bubble', 512, 128)
    cv2.imshow('bubble', bubble)

    cv2.namedWindow("residue", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('residue', 512, 128)
    cv2.imshow('residue', residue)

    cv2.namedWindow("prediction_bubble", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('prediction_bubble', 512, 128)
    cv2.imshow('prediction_bubble', prediction_bubble)

    cv2.namedWindow("prediction_residue", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('prediction_residue', 512, 128)
    cv2.imshow('prediction_residue', prediction_residue)

    cv2.waitKey(100)

print('training finished')
print('top accuracy = ', top_accuracy)

