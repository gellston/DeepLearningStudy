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


training_epochs = 300
batch_size = 30
target_accuracy = 0.95
learning_rate = 0.0003



datasets = TorchSegmentationDatasetLoaderV2(root_path="D://프로젝트//시약검사//이미지//20230601_3cc 고무_크롭_Segmentation_전처리//",
                                            image_height=416,
                                            image_width=160,
                                            classNum=2,
                                            skipClass=[0],
                                            isColor=False,
                                            isNorm=False)

data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)


BiSegNet = BiSeNetMobileV2(class_num=1,
                           activation=torch.nn.ReLU6).to(device)
print('==== model info ====')
summary(BiSegNet, (1, 416, 160))
BiSegNet.eval()
compiled_model = torch.jit.script(BiSegNet)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//BiSegNet(ReagentV1)_no_train.pt")
print('====================')


loss_fn = nn.BCELoss().to(device)# 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = torch.optim.Adam(BiSegNet.parameters(), lr=learning_rate)

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

    BiSegNet.eval()
    compiled_model = torch.jit.script(BiSegNet)
    torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//BiSegNet(ReagentV1)_step.pt")

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
        torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//BiSegNet(ReagentV1)_top.pt")


    if average_accuracy >= target_accuracy:
        BiSegNet.eval()
        compiled_model = torch.jit.script(BiSegNet)
        torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//BiSegNet(ReagentV1)_final.pt")
        break


    original_image = X[0].permute(1, 2, 0).numpy().astype('uint8')
    original_image = original_image.squeeze(2)

    label_image = Y[0].permute(1, 2, 0).detach().cpu().numpy()
    label_image = np.where(label_image > 0.5, 255, 0).astype('uint8')
    label_image = label_image.squeeze(2)


    prediction_image = prediction[0].permute(1, 2, 0).detach().cpu().numpy()
    prediction_image = np.where(prediction_image > 0.5, 255, 0).astype('uint8')
    prediction_image = prediction_image.squeeze(2)

    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original', 160, 416)
    cv2.imshow('original', original_image)

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('label', 160, 416)
    cv2.imshow('label', label_image)
    
    cv2.namedWindow("prediction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('prediction', 160, 416)
    cv2.imshow('prediction', prediction_image)

    cv2.waitKey(100)

print('training finished')


