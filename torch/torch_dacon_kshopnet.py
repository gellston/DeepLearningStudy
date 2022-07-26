import torch
import torch.nn.functional as F
import random
import math

from util.DaconKshopNetDataset import DaconKshopNetDataset
from torch.utils.data import DataLoader

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


#Hyper parameter
batch_size = 1
#Hyper parameter


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
data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
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
    avg_fuel = (X[0][5].item() / data_count) + avg_fuel
    avg_promotion1 = (X[0][6].item() / data_count) + avg_promotion1
    avg_promotion2 = (X[0][7].item() / data_count) + avg_promotion2
    avg_promotion3 = (X[0][8].item() / data_count) + avg_promotion3
    avg_promotion4 = (X[0][9].item() / data_count) + avg_promotion4
    avg_promotion5 = (X[0][10].item() / data_count) + avg_promotion5
    avg_unemployment = (X[0][11].item() / data_count) + avg_unemployment

    if eps > abs(X[0][5]) and abs(X[0][5]) != 0: eps = abs(X[0][5]).item()
    if eps > abs(X[0][6]) and abs(X[0][6]) != 0: eps = abs(X[0][6]).item()
    if eps > abs(X[0][7]) and abs(X[0][7]) != 0: eps = abs(X[0][7]).item()
    if eps > abs(X[0][8]) and abs(X[0][8]) != 0: eps = abs(X[0][8]).item()
    if eps > abs(X[0][9]) and abs(X[0][9]) != 0: eps = abs(X[0][9]).item()
    if eps > abs(X[0][10]) and abs(X[0][10]) != 0: eps = abs(X[0][10]).item()
    if eps > abs(X[0][11]) and abs(X[0][11]) != 0: eps = abs(X[0][11]).item()

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
    var_fuel = (pow(X[0][5] - avg_fuel, 2) / data_count) + var_fuel
    var_promotion1 = (pow(X[0][6] - avg_promotion1, 2) / data_count) + var_promotion1
    var_promotion2 = (pow(X[0][7] - avg_promotion2, 2) / data_count) + var_promotion2
    var_promotion3 = (pow(X[0][8] - avg_promotion3, 2) / data_count) + var_promotion3
    var_promotion4 = (pow(X[0][9] - avg_promotion4, 2) / data_count) + var_promotion4
    var_promotion5 = (pow(X[0][10] - avg_promotion5, 2) / data_count) + var_promotion5
    var_unemployment = (pow(X[0][11] - avg_unemployment, 2) / data_count) + var_unemployment

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