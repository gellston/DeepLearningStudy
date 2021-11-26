import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from model.SoftmaxClassifierModel import SoftmaxClassifierModel

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]

y_train = [2, 2, 2, 1, 1, 1, 0, 0]

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
print(x_train.shape)
print(y_train.shape)

y_one_hot = torch.zeros(8, 3)#heightxwidth
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print(y_one_hot.shape)
print(y_one_hot)


# 모델 초기화
model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):

    # 가설
    prediction = model(x_train)

    # 비용 함수
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))


print('==== result ====')
result = F.softmax(model(x_train), dim=1)
result = result > 0.5
print(result)
print('==== result ====')
