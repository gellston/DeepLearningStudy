import torch
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

from util.DaconKshopNetDataset import DaconKshopNetDataset
from torch.utils.data import DataLoader
from model.KShopNetV3 import KShopNetV3



USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)



#Training Start
class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class RMSELLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        #self.eps = eps

    def forward(self, yhat, y):

        yhat = torch.log1p(yhat)
        y = torch.log1p(y)
        loss = torch.sqrt(self.mse(yhat, y))
        return loss


#Hyper parameter
#batch_size = data_count
train_batch_size = 6255
test_batch_size = 1
training_epochs = 6000
learning_rate = 0.09
print('final calculate learning rate =', learning_rate)
#Hyper parameter

train_datasets = DaconKshopNetDataset(train_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//train.csv',
                                      test_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//test.csv',
                                      ops='train',
                                      norm=False,
                                      d2shape=True,
                                      eps=0,
                                      average=[],
                                      stdev=[])

train_data_loader = DataLoader(train_datasets,
                               batch_size=train_batch_size,
                               shuffle=True,
                               drop_last=True)

model = KShopNetV3(activation=torch.nn.SiLU,
                   expand_rate=0.5,
                   dropout_rate=0.4).to(device)
model.train()
rms = RMSELoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

plt.rcParams["figure.figsize"] = (12, 8)
figure, axis = plt.subplots(2)

avg_train_graph = []
avg_eval_graph = []
epochs = []

for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_train_total = 0

    train_datasets = torch.utils.data.Subset(train_datasets, torch.randperm(len(train_datasets)))
    total_train_batch = len(train_data_loader)

    for X, Y in train_data_loader:
        gpu_X = X.to(device) #input
        gpu_Y = Y.to(device) #output

        model.train()
        optimizer.zero_grad()
        hypothesis = model(gpu_X)

        rms_cost = rms(hypothesis, gpu_Y)
        rms_cost.backward()

        avg_train_total += (rms_cost / total_train_batch)
        optimizer.step()

    if avg_train_total < 290000:
        break

    avg_train_graph.append(avg_train_total.cpu().detach().numpy())
    epochs.append(epoch)

    plt.show(block=False)
    plt.pause(0.001)
    axis[0].plot(epochs, avg_train_graph)
    axis[0].set_title("(TRAIN RMS)")
    plt.show(block=False)
    plt.pause(0.001)
    print('Epoch:', '%04d' % (epoch + 1), '(TRAIN RMS) = {:.9f}'.format(avg_train_total))


#Model Save
plt.savefig('C://Github//DeepLearningStudy//trained_model//KShopNetResult.png')


#model.eval()
#compiled_model = torch.jit.script(model)
#torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//KShopNetV2.pt")










##Prediction Start!!!!!
datasets = DaconKshopNetDataset(train_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//train.csv',
                                test_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//test.csv',
                                ops='test',
                                norm=False,
                                d2shape=True,
                                eps=0,
                                average=[],
                                stdev=[])

data_loader = DataLoader(datasets,
                         batch_size=1,
                         shuffle=False,
                         drop_last=False)

submission_file = pd.read_csv("C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//sample_submission.csv")
test_result = []
for X, Y in data_loader:
    gpu_X = X.to(device)  # input
    gpu_Y = Y.to(device)  # output

    model.eval()
    hypothesis = model(gpu_X)
    test_result.append(hypothesis[0].cpu().detach().numpy().item())
    print('result =', hypothesis[0])

submission_file["Weekly_Sales"] = test_result
submission_file.to_csv("C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//sample_submission_result.csv")
##Prediction Start!!!!!