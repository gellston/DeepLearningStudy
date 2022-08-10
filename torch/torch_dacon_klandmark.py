import torch
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision
from torchsummary import summary

from torch.utils.data import Dataset


from util.DaconKLandMarkDataset import DaconKLandMarkDataset
from torch.utils.data import DataLoader
from model.KLandMarkNet import KLandMarkNet18
from util.nf_helper import AGC



USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)




#Hyper parameter
#batch_size = data_count
batch_size = 10
training_epochs = 100
learning_rate = 0.003
target_accuracy = 0.95
#Hyper parameter



datasets = DaconKLandMarkDataset(data_root='D://Github//DeepLearningStudy//dataset//dacon_klandmark//',
                                 image_width=448,
                                 image_height=448,
                                 ops='train')


data_loader = DataLoader(datasets,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)


model = KLandMarkNet18(class_num=10,
                       block_dropout_probability=0.3,
                       gap_dropout_probability=0.3,
                       stochastic_probability=0.3).to(device)
print('==== model info ====')
summary(model, (3, 448, 448))
print('====================')

model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "D://Github//DeepLearningStudy//trained_model//KLandMarkNet18.pt")

model.train()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate * batch_size / 255)
optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['fc'], clipping=0.1)


plt.rcParams["figure.figsize"] = (12, 8)
figure, axis = plt.subplots(2)

avg_cost_graph = []
avg_acc_graph = []
epochs = []



for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
    total_batch = len(data_loader)
    datasets = torch.utils.data.Subset(datasets, torch.randperm(len(datasets)))
    for X, Y in data_loader:
        gpu_X = X.to(device).detach()
        gpu_Y = Y.to(device).detach()


        #check_image = X[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
        #cv2.namedWindow("input_image", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('input_image', 448, 448)
        #cv2.imshow('input_image', check_image)
        #cv2.waitKey(10)

        model.train()
        optimizer.zero_grad()
        hypothesis = model(gpu_X)
        cost = criterion(hypothesis, gpu_Y)
        cost.backward()
        avg_cost += (cost / total_batch)
        optimizer.step()

        model.eval()
        prediction = model(gpu_X)
        correct_prediction = torch.argmax(prediction, 1) == torch.argmax(gpu_Y, 1)
        accuracy = correct_prediction.float().mean()
        avg_acc += (accuracy / total_batch)


    avg_acc_graph.append(avg_acc.cpu().detach().numpy())
    avg_cost_graph.append(avg_cost.cpu().detach().numpy())
    epochs.append(epoch)


    axis[0].plot(epochs, avg_acc_graph)
    axis[0].set_title("accuracy")
    axis[1].plot(epochs, avg_cost_graph)
    axis[1].set_title("cost")
    plt.show(block=False)
    plt.pause(0.001)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > target_accuracy:
        break

plt.savefig('D://Github//DeepLearningStudy//trained_model//KLandResult.png')


model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "D://Github//DeepLearningStudy//trained_model//KLandMarkNet18_trained.pt")



test_datasets = DaconKLandMarkDataset(data_root='D://Github//DeepLearningStudy//dataset//dacon_klandmark//',
                                      image_width=448,
                                      image_height=448,
                                      ops='test')
test_data_loader = DataLoader(test_datasets,
                              batch_size=1,
                              shuffle=False,
                              drop_last=False)

submission_file = pd.read_csv("D://Github//DeepLearningStudy//dataset//dacon_klandmark//sample_submission.csv")
test_result = []
count = 0
for X in test_data_loader:
    gpu_X = X.to(device)  # input

    model.eval()
    hypothesis = model(gpu_X)
    hypothesis = torch.argmax(hypothesis, dim=1)
    label = hypothesis[0].cpu().detach().numpy().item()
    test_result.append(label)
    print('result =', hypothesis[0])

submission_file["label"] = test_result
submission_file.to_csv("D://Github//DeepLearningStudy//dataset//dacon_klandmark//sample_submission_result.csv", index=False)