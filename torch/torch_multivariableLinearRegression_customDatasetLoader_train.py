import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from model.MultivariateLinearRegressionModel import MultivariateLinearRegressionModel
from util.TorchBasicCustomDataLoader import TorchBasicCustomDataLoader


model = MultivariateLinearRegressionModel()
print(list(model.parameters()))

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

dataset = TorchBasicCustomDataLoader()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):

        x_train, y_train = samples

        # H(x) 계산
        prediction = model(x_train)
        # model(x_train)은 model.forward(x_train)와 동일함.

        # cost 계산
        cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

        # cost로 H(x) 개선하는 부분
        # gradient를 0으로 초기화
        optimizer.zero_grad()
        # 비용 함수를 미분하여 gradient 계산
        cost.backward()
        # W와 b를 업데이트
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()))

# 임의의 입력 [73, 80, 75]를 선언
new_var = torch.FloatTensor([[73, 80, 75]])
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)