import torch
import torch.nn as nn
import random
import cv2
import numpy as np


from torch.utils.data import DataLoader


from util.FIATClassificationDataset import FIATClassificationDataset


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


## Hyper parameter
training_epochs = 1
batch_size = 10
target_accuracy = 0.99
learning_rate = 0.003
accuracy_threshold = 0.5
## Hyper parameter




model = torch.jit.load("C://Github//DeepLearningStudy//trained_model//food_test.pt").to(device)
model.eval()



## no Train Model Save
datasets = FIATClassificationDataset('C://Github//DeepLearningStudy//dataset//FIAT_dataset_food//',
                                     label_height=224,
                                     label_width=224,
                                     isColor=True,
                                     isNorm=False)
data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)


for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        gpu_X = X.to(device)
        gpu_Y = Y.to(device)

        X_single_data = gpu_X[0].to(device)
        Y_single_data = gpu_Y[0].to(device)
        X_single_data = X_single_data.unsqueeze(0).float()
        Y_single_data = Y_single_data.unsqueeze(0).float()

        print('x shape =', X_single_data.shape)
        print('y shape =', Y_single_data.shape)

        single_prediction = model(X_single_data)
        print('prediction shape =', single_prediction.shape)
        print('Label Prediction: ', torch.argmax(single_prediction, 1).item())
        test_image = X_single_data.permute(0, 2, 3, 1).squeeze(0).cpu().numpy().astype(np.uint8)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', 512, 512)
        cv2.imshow('result', test_image)
        cv2.waitKey(1000)



print('prediction finished')