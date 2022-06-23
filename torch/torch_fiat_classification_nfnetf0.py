import torch
import torch.nn as nn
import random

from ptflops import get_model_complexity_info
from torchsummary import summary
from torch.utils.data import DataLoader

from model.NFNet import NFNetF0
from util.FIATClassificationDataset import FIATClassificationDataset
from util.nf_helper import AGC

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


## Hyper parameter
training_epochs = 30
batch_size = 22
target_accuracy = 0.99
learning_rate = 0.003
accuracy_threshold = 0.5
image_width = 224
image_height = 224
## Hyper parameter


model = NFNetF0(class_num=4,
                gap_dropout_probability=0.25,
                stochastic_probability=0.25).to(device)

print('==== model info ====')
summary(model, (3, image_height, image_width))
print('====================')


macs, params = get_model_complexity_info(model,
                                         (3, image_height, image_width),
                                         as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


## no Train Model Save
model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//FIAT(NFNetF0).pt")
## no Train Model Save


datasets = FIATClassificationDataset('C://Github//DeepLearningStudy//dataset//FIAT_dataset_food//',
                                     label_height=image_height,
                                     label_width=image_width,
                                     isColor=True,
                                     isNorm=False)
data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)


model.train()
criterion = nn.BCELoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate * batch_size / 255)
optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['fc'], clipping=0.1)

for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        gpu_X = X.to(device)
        gpu_Y = Y.to(device)

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

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > target_accuracy:
        break

## no Train Model Save
model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//FIAT(NFNetF0).pt")
## no Train Model Save

print('Learning finished')