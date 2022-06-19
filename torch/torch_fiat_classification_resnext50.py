import torch
import torch.nn as nn

from ptflops import get_model_complexity_info
from torchsummary import summary
from torch.utils.data import DataLoader

from model.ResNextnet50 import ResNextnet50
from util.FIATClassificationDataset import FIATClassificationDataset


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


## Hyper parameter
training_epochs = 30
batch_size = 15
target_accuracy = 0.99
learning_rate = 0.0001
accuracy_threshold = 0.5
## Hyper parameter


model = ResNextnet50(class_num=4,
                     activation=torch.nn.ReLU,
                     groups=32).to(device)
print('==== model info ====')
summary(model, (3, 224, 224))
print('====================')

macs, params = get_model_complexity_info(model,
                                         (3, 224, 224),
                                         as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


#weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
model.apply(init_weights)


## no Train Model Save

model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//FIAT(ResNextnet50).pt")

## no Train Model Save


datasets = FIATClassificationDataset('C://Github//DeepLearningStudy//dataset//FIAT_dataset_food//',
                                     label_height=224,
                                     label_width=224,
                                     isColor=True,
                                     isNorm=False)
data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)


model.train()
criterion = nn.BCELoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

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
        break;

## no Train Model Save
model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//TRAIN_FIAT(ResNextnet50).pt")
## no Train Model Save

print('Learning finished')