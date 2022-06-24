import torch
import torch.nn as nn
import random
import torchvision
import gc
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

from ptflops import get_model_complexity_info
from torchsummary import summary
from torch.utils.data import DataLoader
from util.nf_helper import AGC
from model.NFNet import NFNetF0

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


## Hyper parameter
training_epochs = 40
current_epoch = 0
batch_size = 5
target_accuracy = 0.80
learning_rate = 0.003
num_class = 1000
pretrained = False
## Hyper parameter


model = NFNetF0(class_num=num_class,
                gap_dropout_probability=0.25,
                stochastic_probability=0.25).to(device)
print('==== model info ====')
summary(model, (3, 224, 224))
print('====================')


macs, params = get_model_complexity_info(model,
                                         (3, 224, 224),
                                         as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


## no Train Model Save
if pretrained == True:
    CSPMobileNetV2Weight = torch.jit.load("D://Github//DeepLearningStudy//trained_model//ImageNet(NFNetF0).pt")
    model.load_state_dict(CSPMobileNetV2Weight.state_dict())

model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "D://Github//DeepLearningStudy//trained_model//ImageNet(NFNetF0).pt")
## no Train Model Save


transform = torchvision.transforms.Compose([
                #torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize((640, 640)),
                torchvision.transforms.ToTensor()
            ])

trainDataset = torchvision.datasets.ImageNet(root="D://학습이미지//imagenet//",
                                                      split='val',
                                                      transform=transform)

validationDataset = torchvision.datasets.ImageNet(root="D://학습이미지//imagenet//",
                                                      split='val',
                                                      transform=transform)

# dataset loader
train_data_loader = DataLoader(dataset=trainDataset,
                               batch_size=batch_size,  # 배치 크기는 100
                               shuffle=True,
                               drop_last=True)

validation_data_loader = DataLoader(dataset=validationDataset,
                                    batch_size=1,  # 배치 크기는 100
                                    shuffle=True,
                                    drop_last=True)

model.train()
criterion = nn.BCELoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['fc'], clipping=0.01)


avg_train_cost_graph = []
avg_train_acc_graph = []
avg_val_cost_graph = []
avg_val_acc_graph = []
epoches = []


plt.rcParams["figure.figsize"] = (12, 8)
figure, axis = plt.subplots(2, 2)

for epoch in range(current_epoch, training_epochs): # 앞서 training_epochs의 값은 15로 지정함.

    avg_train_cost = 0
    avg_train_acc = 0

    avg_validation_cost = 0
    avg_validation_acc = 0

    total_train_batch = len(train_data_loader)
    print('train total_batch = ', total_train_batch)

    total_validation_batch = len(validation_data_loader)
    print('validation total_batch = ', total_validation_batch)

    current_batch = 0
    for X, Y in train_data_loader:
        gpu_X = X.to(device)
        gpu_Y = Y.to(device)
        gpu_Y = torch.nn.functional.one_hot(gpu_Y, num_classes=num_class).float()

        model.train()
        optimizer.zero_grad()
        hypothesis = model(gpu_X)
        cost = criterion(hypothesis, gpu_Y)
        cost.backward()
        avg_train_cost += (cost / total_train_batch)
        optimizer.step()

        model.eval()
        hypothesis = model(gpu_X)
        correct_prediction = torch.argmax(hypothesis, 1) == torch.argmax(gpu_Y, 1)
        accuracy = correct_prediction.float().mean()
        avg_train_acc += (accuracy / total_train_batch)

        current_batch += 1
        print('current train batch=', current_batch, '/', total_train_batch, 'epoch=', epoch, ' accuracy=', accuracy.item(), ' cost=', cost.item())

    gc.collect()
    current_batch = 0
    for X, Y in validation_data_loader:
        gpu_X = X.to(device)
        gpu_Y = Y.to(device)
        gpu_Y = torch.nn.functional.one_hot(gpu_Y, num_classes=num_class).float()

        model.eval()
        hypothesis = model(gpu_X)
        cost = criterion(hypothesis, gpu_Y)
        avg_validation_cost += (cost / total_validation_batch)

        model.eval()
        hypothesis = model(gpu_X)
        correct_prediction = torch.argmax(hypothesis, 1) == torch.argmax(gpu_Y, 1)
        accuracy = correct_prediction.float().mean()
        avg_validation_acc += (accuracy / total_validation_batch)

        current_batch += 1
        print('current validation batch=', current_batch, '/', total_validation_batch, 'epoch=', epoch, ' accuracy=', accuracy.item(), ' cost=', cost.item())

    avg_train_acc_graph.append(avg_train_acc)
    avg_train_cost_graph.append(avg_train_cost)
    avg_train_acc_graph.append(avg_validation_acc)
    avg_train_cost_graph.append(avg_validation_cost)
    epoches.append(epoch)

    model.eval()
    compiled_model = torch.jit.script(model)
    torch.jit.save(compiled_model, "D://Github//DeepLearningStudy//trained_model//ImageNet(NFNetF0).pt")
    print('Train Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    print('Validation Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_validation_cost), 'acc =', '{:.9f}'.format(avg_validation_acc))


    plt.show(block=False)
    plt.pause(0.001)
    axis[0, 0].plot(epoches, avg_train_cost_graph)
    axis[0, 0].set_title("train cost")
    axis[0, 1].plot(epoches, avg_val_cost_graph)
    axis[0, 1].set_title("validation cost")
    axis[1, 0].plot(epoches, avg_train_acc_graph)
    axis[1, 0].set_title("train accuracy")
    axis[1, 0].set_yscale('linear')
    axis[1, 0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    axis[1, 1].plot(epoches, avg_val_acc_graph)
    axis[1, 1].set_title("validation accuracy")
    axis[1, 1].set_yscale('linear')
    axis[1, 1].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.show(block=False)
    plt.pause(0.001)
    plt.savefig('D://Github//DeepLearningStudy//trained_model//ImageNet(NFNetF0).png')

    if avg_validation_acc > target_accuracy:
        break

## no Train Model Save
model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "D://Github//DeepLearningStudy//trained_model//ImageNet(NFNetF0).pt")
## no Train Model Save

print('Learning finished')