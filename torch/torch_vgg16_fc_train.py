import torch
import torch.nn as nn
import random

from torchsummary import summary
from torch.utils.data import DataLoader

from util.TorchClassifierDatasetLoader import TorchClassifierDatasetLoader
from model.VGG16FC import VGG16FC

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)







# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


training_epochs = 15
batch_size = 5


datasets = TorchClassifierDatasetLoader('C://Github//DeepLearningStudy//dataset//animal-train-v1', image_height=224, image_width=224)
data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)


VGG16 = VGG16FC(class_num=5).to(device)
print('==== model info ====')
summary(VGG16, (3, 224, 224))
print('====================')


VGG16.eval()
compiled_model = torch.jit.script(VGG16)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//NoVgg16.pt")
## no Train Model Save




loss_fn = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = torch.optim.SGD(VGG16.parameters(), lr=0.003)


total_batch = len(data_loader)
print('total_batch=', total_batch)

final_cost = 0
final_accuracy = 0

for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        ##cost calculation
        VGG16.train()
        optimizer.zero_grad()
        hypothesis = VGG16(X)
        cost = loss_fn(hypothesis, Y)
        cost.backward()
        optimizer.step()

        ##acc calculation
        VGG16.eval()
        prediction = VGG16(X)
        correct_prediction = torch.argmax(prediction, 1) == Y
        accuracy = correct_prediction.float().mean()

        avg_cost += (cost / total_batch)
        avg_acc += (accuracy / total_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > 0.9:
        final_accuracy = avg_acc
        final_cost = avg_acc
        break;

print('Final accuracy = ', final_accuracy, ', cost = ', final_cost)
print('Learning finished')




