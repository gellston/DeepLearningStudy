import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



#hyperparaemter
learnig_rate = 0.003
nb_epochs = 1000
#hyperparaemter


x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
print(x_train.shape)
print(y_train.shape)


model = nn.Sequential(
   nn.Linear(2, 1), # input_dim = 2, output_dim = 1
   nn.Sigmoid() # 출력은 시그모이드 함수를 거친다
)

print(model(x_train))


optimizer = optim.SGD(model.parameters(), lr=learnig_rate)


for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = model(x_train)
    #cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))


print(model(x_train))


prediction = model(x_train) >= torch.FloatTensor([0.5])
print(prediction)

