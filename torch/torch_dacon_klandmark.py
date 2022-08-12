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
target_accuracy = 0.99
image_width = 480
image_height = 320
#Hyper parameter



augmentation_dataset = DaconKLandMarkDataset(data_root='C://Github//DeepLearningStudy//dataset//dacon_klandmark//',
                                             image_width=image_width,
                                             image_height=image_height,
                                             ops='train')


augmentation_data_loader = DataLoader(augmentation_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True)

resize_augmentation_dataset = DaconKLandMarkDataset(data_root='C://Github//DeepLearningStudy//dataset//dacon_klandmark//',
                                                    image_width=240,
                                                    image_height=160,
                                                    no_augmentation=True,
                                                    ops='train')


resize_augmentation_data_loader = DataLoader(resize_augmentation_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=True)


no_augmentation_dataset = DaconKLandMarkDataset(data_root='C://Github//DeepLearningStudy//dataset//dacon_klandmark//',
                                                image_width=image_width,
                                                image_height=image_height,
                                                no_augmentation=True,
                                                ops='train')


no_augmentation_data_loader = DataLoader(no_augmentation_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True)


model = KLandMarkNet18(class_num=10,
                       gap_dropout_probability=0.35,
                       block_dropout_probability=0.35,
                       stochastic_probability=0.35).to(device)


print('==== model info ====')
summary(model, (3, image_height, image_width))
print('====================')

model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//KLandMarkNet18.pt")

model.train()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate * batch_size / 255)
optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['fc'], clipping=0.1)


plt.rcParams["figure.figsize"] = (12, 12)
figure, axis = plt.subplots(2, 4)


augmentation_avg_cost_graph = []
augmentation_avg_acc_graph = []

no_augmentation_avg_cost_graph = []
no_augmentation_avg_acc_graph = []

resize_augmentation_avg_cost_graph = []
resize_augmentation_avg_acc_graph = []

epochs = []


for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.

    augmentation_avg_cost = 0
    augmentation_avg_acc = 0

    no_augmentation_avg_cost = 0
    no_augmentation_avg_acc = 0

    resize_augmentation_avg_cost = 0
    resize_augmentation_avg_acc = 0

    total_batch = len(augmentation_data_loader)
    augmentation_dataset = torch.utils.data.Subset(augmentation_dataset, torch.randperm(len(augmentation_dataset)))
    no_augmentation_dataset = torch.utils.data.Subset(no_augmentation_dataset, torch.randperm(len(no_augmentation_dataset)))
    resize_augmentation_dataset = torch.utils.data.Subset(resize_augmentation_dataset, torch.randperm(len(resize_augmentation_dataset)))

    print('augmentation dataset train start')
    for X, Y in augmentation_data_loader:
        gpu_X = X.to(device).detach()
        gpu_Y = Y.to(device).detach()

        check_image = X[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
        cv2.namedWindow("input_image", cv2.WINDOW_NORMAL)
        cv2.imshow('input_image', check_image)
        cv2.waitKey(10)

        model.train()
        optimizer.zero_grad()
        hypothesis = model(gpu_X)
        cost = criterion(hypothesis, gpu_Y)
        cost.backward()
        augmentation_avg_cost += (cost / total_batch)
        optimizer.step()

        model.eval()
        prediction = model(gpu_X)
        correct_prediction = torch.argmax(prediction, 1) == torch.argmax(gpu_Y, 1)
        accuracy = correct_prediction.float().mean()
        augmentation_avg_acc += (accuracy / total_batch)

    print('no augmentation dataset train start')
    for X, Y in no_augmentation_data_loader:
        gpu_X = X.to(device).detach()
        gpu_Y = Y.to(device).detach()

        check_image = X[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
        cv2.namedWindow("input_image", cv2.WINDOW_NORMAL)
        cv2.imshow('input_image', check_image)
        cv2.waitKey(10)

        model.train()
        optimizer.zero_grad()
        hypothesis = model(gpu_X)
        cost = criterion(hypothesis, gpu_Y)
        cost.backward()
        no_augmentation_avg_cost += (cost / total_batch)
        optimizer.step()

        model.eval()
        prediction = model(gpu_X)
        correct_prediction = torch.argmax(prediction, 1) == torch.argmax(gpu_Y, 1)
        accuracy = correct_prediction.float().mean()
        no_augmentation_avg_acc += (accuracy / total_batch)

    print('resize augmentation dataset train start')
    for X, Y in resize_augmentation_data_loader:
        gpu_X = X.to(device).detach()
        gpu_Y = Y.to(device).detach()

        check_image = X[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
        cv2.namedWindow("input_image", cv2.WINDOW_NORMAL)
        cv2.imshow('input_image', check_image)
        cv2.waitKey(10)

        model.train()
        optimizer.zero_grad()
        hypothesis = model(gpu_X)
        cost = criterion(hypothesis, gpu_Y)
        cost.backward()
        resize_augmentation_avg_cost += (cost / total_batch)
        optimizer.step()

        model.eval()
        prediction = model(gpu_X)
        correct_prediction = torch.argmax(prediction, 1) == torch.argmax(gpu_Y, 1)
        accuracy = correct_prediction.float().mean()
        resize_augmentation_avg_acc += (accuracy / total_batch)


    augmentation_avg_acc_graph.append(augmentation_avg_acc.cpu().detach().numpy())
    augmentation_avg_cost_graph.append(augmentation_avg_cost.cpu().detach().numpy())
    no_augmentation_avg_acc_graph.append(no_augmentation_avg_acc.cpu().detach().numpy())
    no_augmentation_avg_cost_graph.append(no_augmentation_avg_cost.cpu().detach().numpy())
    resize_augmentation_avg_acc_graph.append(resize_augmentation_avg_acc.cpu().detach().numpy())
    resize_augmentation_avg_cost_graph.append(resize_augmentation_avg_cost.cpu().detach().numpy())

    epochs.append(epoch)


    axis[0, 0].plot(epochs, augmentation_avg_acc_graph)
    axis[0, 0].set_title("augmentation accuracy")
    axis[0, 1].plot(epochs, augmentation_avg_cost_graph)
    axis[0, 1].set_title("augmentation cost")
    axis[0, 2].plot(epochs, no_augmentation_avg_acc_graph)
    axis[0, 2].set_title("no_augmentation accuracy")
    axis[0, 3].plot(epochs, no_augmentation_avg_cost_graph)
    axis[0, 3].set_title("no_augmentation cost")
    axis[1, 0].plot(epochs, resize_augmentation_avg_acc_graph)
    axis[1, 0].set_title("resize_augmentation accuracy")
    axis[1, 1].plot(epochs, resize_augmentation_avg_cost_graph)
    axis[1, 1].set_title("resize_augmentation cost")
    plt.show(block=False)
    plt.pause(0.001)

    print('Epoch:', '%04d' % (epoch + 1), 'aug cost =', '{:.9f}'.format(augmentation_avg_cost),
                                          'aug acc =', '{:.9f}'.format(augmentation_avg_acc),
                                          'no aug cost =', '{:.9f}'.format(no_augmentation_avg_cost),
                                          'no aug acc =', '{:.9f}'.format(no_augmentation_avg_acc),
                                          'resize aug cost =', '{:.9f}'.format(resize_augmentation_avg_cost),
                                          'resize aug acc =', '{:.9f}'.format(resize_augmentation_avg_acc))
    if augmentation_avg_acc >= 0.95 and no_augmentation_avg_acc >= target_accuracy and resize_augmentation_avg_acc >= target_accuracy:
        break

plt.savefig('D://Github//DeepLearningStudy//trained_model//KLandResult.png')

model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "D://Github//DeepLearningStudy//trained_model//KLandMarkNet18_trained.pt")






test_datasets = DaconKLandMarkDataset(data_root='C://Github//DeepLearningStudy//dataset//dacon_klandmark//',
                                      image_width=image_width,
                                      image_height=image_height,
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

    check_image = X[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
    cv2.namedWindow("input_image", cv2.WINDOW_NORMAL)
    cv2.imshow('input_image', check_image)
    cv2.waitKey(10)

    model.eval()
    hypothesis = model(gpu_X)
    hypothesis = torch.argmax(hypothesis, dim=1)
    label = hypothesis[0].cpu().detach().numpy().item()
    test_result.append(label)
    print('result =', hypothesis[0])

submission_file["label"] = test_result
submission_file.to_csv("C://Github//DeepLearningStudy//dataset//dacon_klandmark//sample_submission_result.csv", index=False)