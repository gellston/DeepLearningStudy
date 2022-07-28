import torch
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

from util.DaconKshopNetDataset import DaconKshopNetDataset
from torch.utils.data import DataLoader
from model.KShopNet import KShopNet



USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)




# 0 store
# 1 day
# 2 month
# 3 year
# 4 temperature
# 5 Fuel Price  표준편차 이용
# 6 Promotion1  표준편차 이용
# 7 Promotion2  표준편차 이용
# 8 Promotion3  표준편차 이용
# 9 Promotion4  표준편차 이용
# 10 Promotion5 표준편차 이용
# 11 Unemployment 표준편차 이용
# 12 Holiyday
datasets = DaconKshopNetDataset(train_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//train.csv',
                                test_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//test.csv',
                                ops='train',
                                norm=False)
data_loader = DataLoader(datasets, batch_size=1, shuffle=True)
data_count = len(data_loader)

eps = 100
avg_fuel = 0
avg_promotion1 = 0
avg_promotion2 = 0
avg_promotion3 = 0
avg_promotion4 = 0
avg_promotion5 = 0
avg_unemployment = 0

for X, Y in data_loader:
    avg_fuel = (X[0][4].item() / data_count) + avg_fuel
    avg_promotion1 = (X[0][5].item() / data_count) + avg_promotion1
    avg_promotion2 = (X[0][6].item() / data_count) + avg_promotion2
    avg_promotion3 = (X[0][7].item() / data_count) + avg_promotion3
    avg_promotion4 = (X[0][8].item() / data_count) + avg_promotion4
    avg_promotion5 = (X[0][9].item() / data_count) + avg_promotion5
    avg_unemployment = (X[0][10].item() / data_count) + avg_unemployment

    if eps > abs(X[0][4]) and abs(X[0][4]) != 0: eps = abs(X[0][4]).item()
    if eps > abs(X[0][5]) and abs(X[0][5]) != 0: eps = abs(X[0][5]).item()
    if eps > abs(X[0][6]) and abs(X[0][6]) != 0: eps = abs(X[0][6]).item()
    if eps > abs(X[0][7]) and abs(X[0][7]) != 0: eps = abs(X[0][7]).item()
    if eps > abs(X[0][8]) and abs(X[0][8]) != 0: eps = abs(X[0][8]).item()
    if eps > abs(X[0][9]) and abs(X[0][9]) != 0: eps = abs(X[0][9]).item()
    if eps > abs(X[0][10]) and abs(X[0][10]) != 0: eps = abs(X[0][10]).item()

eps = eps / 100
print('epsilon = ', eps)

print('==== average ====')
print('avg_fuel = ', avg_fuel)
print('avg_promotion1 = ', avg_promotion1)
print('avg_promotion2 = ', avg_promotion2)
print('avg_promotion3 = ', avg_promotion3)
print('avg_promotion4 = ', avg_promotion4)
print('avg_promotion5 = ', avg_promotion5)
print('avg_unemployment = ', avg_unemployment)
print('==================')


var_fuel = 0
var_promotion1 = 0
var_promotion2 = 0
var_promotion3 = 0
var_promotion4 = 0
var_promotion5 = 0
var_unemployment = 0

for X, Y in data_loader:
    var_fuel = (pow(X[0][4] - avg_fuel, 2) / data_count) + var_fuel
    var_promotion1 = (pow(X[0][5] - avg_promotion1, 2) / data_count) + var_promotion1
    var_promotion2 = (pow(X[0][6] - avg_promotion2, 2) / data_count) + var_promotion2
    var_promotion3 = (pow(X[0][7] - avg_promotion3, 2) / data_count) + var_promotion3
    var_promotion4 = (pow(X[0][8] - avg_promotion4, 2) / data_count) + var_promotion4
    var_promotion5 = (pow(X[0][9] - avg_promotion5, 2) / data_count) + var_promotion5
    var_unemployment = (pow(X[0][10] - avg_unemployment, 2) / data_count) + var_unemployment

print('==== variance ====')
print('var_fuel = ', var_fuel)
print('var_promotion1 = ', var_promotion1)
print('var_promotion2 = ', var_promotion2)
print('var_promotion3 = ', var_promotion3)
print('var_promotion4 = ', var_promotion4)
print('var_promotion5 = ', var_promotion5)
print('var_unemployment = ', var_unemployment)
print('==================')


std_fuel = math.sqrt(var_fuel)
std_promotion1 = math.sqrt(var_promotion1)
std_promotion2 = math.sqrt(var_promotion2)
std_promotion3 = math.sqrt(var_promotion3)
std_promotion4 = math.sqrt(var_promotion4)
std_promotion5 = math.sqrt(var_promotion5)
std_unemployment = math.sqrt(var_unemployment)


print('==== standard deviation ====')
print('std_fuel = ', std_fuel)
print('std_promotion1 = ', std_promotion1)
print('std_promotion2 = ', std_promotion2)
print('std_promotion3 = ', std_promotion3)
print('std_promotion4 = ', std_promotion4)
print('std_promotion5 = ', std_promotion5)
print('std_unemployment = ', std_unemployment)
print('==================')







#Norm parameter input
#Data Checking
stdlist = [std_fuel, std_promotion1, std_promotion2, std_promotion3, std_promotion4, std_promotion5, std_unemployment]
avglist = [avg_fuel, avg_promotion1, avg_promotion2, avg_promotion3, avg_promotion4, avg_promotion5, avg_unemployment]

datasets = DaconKshopNetDataset(train_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//train.csv',
                                test_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//test.csv',
                                ops='train',
                                norm=True,
                                d2shape=True,
                                eps=eps,
                                average=avglist,
                                stdev=stdlist)

data_loader = DataLoader(datasets, batch_size=1, shuffle=True)
data_count = len(data_loader)


#for X, Y in data_loader:
#    print('x shape = ', X.shape)
#    print('y shape = ', Y.shape)
#    print('x = ', X[0])
#    print('y = ', Y[0])

# Data Checking



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
batch_size = 256
training_epochs = 6000
learning_rate = 0.3
print('final calculate learning rate =', learning_rate)
#Hyper parameter


datasets = DaconKshopNetDataset(train_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//train.csv',
                                test_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//test.csv',
                                ops='train',
                                norm=False,
                                d2shape=True,
                                eps=eps,
                                average=avglist,
                                stdev=stdlist)

data_loader = DataLoader(datasets,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)


model = KShopNet(layer_length=3,
                 inner_channel=6,
                 activation=torch.nn.PReLU,
                 se_rate=0.5,
                 exapnd_rate=6).to(device)
model.train()
rms = RMSELoss()
rmsl = RMSELLoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)


plt.rcParams["figure.figsize"] = (12, 8)
figure, axis = plt.subplots(2)

avg_total_graph = []
avg_rms_graph = []
epochs = []

for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.

    avg_total = 0
    avg_rms = 0

    datasets = torch.utils.data.Subset(datasets, torch.randperm(len(datasets)))
    total_batch = len(data_loader)
    for X, Y in data_loader:
        gpu_X = X.to(device) #input
        gpu_Y = Y.to(device) #output

        model.train()
        optimizer.zero_grad()
        hypothesis = model(gpu_X)

        rms_cost = rms(hypothesis, gpu_Y)
        rmsl_cost = rmsl(hypothesis, gpu_Y)

        total_cost = rmsl_cost
        total_cost.backward()

        avg_total += (total_cost / total_batch)
        optimizer.step()
        model.eval()

        avg_rms += (rms_cost / total_batch)


    if avg_rms < 30000:
        break

    avg_total_graph.append(avg_total.cpu().detach().numpy())
    avg_rms_graph.append(avg_rms.cpu().detach().numpy())
    epochs.append(epoch)

    plt.show(block=False)
    plt.pause(0.001)
    axis[0].plot(epochs, avg_total_graph)
    axis[0].set_title("(RMSE / 100) + RMSEL")
    axis[1].plot(epochs, avg_rms_graph)
    axis[1].set_title("RMSE")

    plt.show(block=False)
    plt.pause(0.001)
    plt.savefig('C://Github//DeepLearningStudy//trained_model//KShopNetResult.png')


    print('Epoch:', '%04d' % (epoch + 1), '(RMSE / 100 + RMSEL) =', '{:.9f}'.format(avg_total), 'RMS =', '{:.9f}'.format(avg_rms))
    #Model Save
    model.eval()
    compiled_model = torch.jit.script(model)
    torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//KShopNet.pt")





##Prediction Start!!!!!
datasets = DaconKshopNetDataset(train_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//train.csv',
                                test_root='C://Github//DeepLearningStudy//dataset//dacon_shop_profit//dataset//test.csv',
                                ops='test',
                                norm=False,
                                d2shape=True,
                                eps=eps,
                                average=avglist,
                                stdev=stdlist)

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