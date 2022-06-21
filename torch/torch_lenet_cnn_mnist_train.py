import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import random
import cv2 as cv2

from torch.utils.data import DataLoader
from model.LeNet import LeNet

USE_CUDA = torch.cuda.is_available()  # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu")  # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)
print("Torch version:{}".format(torch.__version__))
print("cuda version: {}".format(torch.version.cuda))
print("cudnn version:{}".format(torch.backends.cudnn.version()))

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# hyperparameters
training_epochs = 100
batch_size = 100
learning_rate = 0.003
target_accuracy = 0.99

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,  # 배치 크기는 100
                         shuffle=True,
                         drop_last=True)

# MNIST data image of shape 28 * 28 = 784
model = LeNet().to(device)

# 비용 함수와 옵티마이저 정의
loss_fn = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = optim.Adam(model.parameters(),  lr=learning_rate)

for epoch in range(training_epochs):  # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # 배치 크기가 100이므로 아래의 연산에서 X는 (100, 28, 28)의 텐서가 된다.
        X = X.to(device)
        # 레이블은 원-핫 인코딩이 된 상태가 아니라 0 ~ 9의 정수.
        Y = Y.to(device)



        ##cost calculation
        model.train()
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = loss_fn(hypothesis, Y)
        cost.backward()
        optimizer.step()

        ##acc calculation
        model.eval()
        prediction = model(X)
        correct_prediction = torch.argmax(prediction, 1) == Y
        accuracy = correct_prediction.float().mean()

        avg_cost += (cost / total_batch)
        avg_acc += (accuracy / total_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > target_accuracy:
        break;

print('Learning finished')



# 테스트 데이터를 사용하여 모델을 테스트한다.
with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    model.eval()

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    for i in range(500):
        r = random.randint(0, len(mnist_test) - 1)
        X_single_data = mnist_test.test_data[r:r + 1].to(device)
        Y_single_data = mnist_test.test_labels[r:r + 1].to(device)
        X_single_data = X_single_data.unsqueeze(0).float()
        Y_single_data = Y_single_data.unsqueeze(0).float()

        single_prediction = model(X_single_data)
        print('Label: ', Y_single_data.item(), '    Prediction: ', torch.argmax(single_prediction, 1).item())

        test_image = mnist_test.test_data[r:r + 1].permute(1, 2, 0).numpy()
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', 512, 512)
        cv2.imshow('result', test_image)
        cv2.waitKey(33)

