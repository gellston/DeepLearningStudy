import torch
import torch.nn as nn
import random
import cv2

from torchsummary import summary
from torch.utils.data import DataLoader

from util.TorchSegmentationDatasetLoaderV1 import TorchSegmentationDatasetLoaderV1
from model.CustomSegmentationV1 import CustomSegmentationV1
from util.helper import IOU

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


training_epochs = 15
batch_size = 5
target_accuracy = 0.9


datasets = TorchSegmentationDatasetLoaderV1('C://Github//DeepLearningStudy//dataset//portrait_segmentation_input256x256',
                                            'C://Github//DeepLearningStudy//dataset//portrait_segmentation_label256x256',
                                            image_height=256,
                                            image_width=256,
                                            isColor=True,
                                            isNorm=False)

data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)


model = CustomSegmentationV1().to(device)
print('==== model info ====')
summary(model, (3, 256, 256))
print('====================')


loss_fn = nn.BCELoss().to(device)# 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)


total_batch = len(data_loader)
print('total_batch=', total_batch)

final_cost = 0
final_accuracy = 0

for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0

    for X, Y in data_loader:
        gpu_X = X.to(device)
        gpu_Y = Y.to(device)

        """
        for index in range(batch_size):

            original_image = X[index].permute(1, 2, 0).numpy()
            label_image = Y[index].permute(1, 2, 0).numpy()


            cv2.namedWindow("original", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('original', 256, 256)
            cv2.imshow('original', original_image)

            cv2.namedWindow("label", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('label', 256, 256)
            cv2.imshow('label', label_image)

            cv2.waitKey(33)
        """

        ##cost calculation
        model.train()
        optimizer.zero_grad()
        hypothesis = model(gpu_X)
        cost = loss_fn(hypothesis, gpu_Y)
        cost.backward()
        optimizer.step()

        ##acc calculation
        model.eval()
        prediction = model(gpu_X)
        accuracy = IOU(gpu_X.detach().cpu().numpy(), prediction.detach().cpu().numpy())

        avg_cost += (cost / total_batch)
        avg_acc += (accuracy / total_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > target_accuracy:
        final_accuracy = avg_acc
        final_cost = avg_acc
        break;

print('Final accuracy = ', final_accuracy, ', cost = ', final_cost)
print('Learning finished')




