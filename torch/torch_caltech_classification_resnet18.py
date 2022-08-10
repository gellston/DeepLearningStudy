import torch
import torch.nn as nn
import random
import torchvision

from ptflops import get_model_complexity_info
from torchsummary import summary
from torch.utils.data import DataLoader

from model.Resnet import ResNet18



USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


## Hyper parameter
training_epochs = 100
batch_size = 20
target_accuracy = 0.90
learning_rate = 0.0001
accuracy_threshold = 0.5
## Hyper parameter


model = ResNet18(class_num=257, activation=torch.nn.SiLU).to(device)
print('==== model info ====')
summary(model, (3, 512, 512))
print('====================')

macs, params = get_model_complexity_info(model,
                                         (3, 512, 512),
                                         as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))




## no Train Model Save

model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//CALTECH256(Resnet18).pt")

trace_input = torch.rand(1, 3, 512, 512).to(device, dtype=torch.float32)
trace_model = torch.jit.trace(model, trace_input)
torch.jit.save(trace_model, "C://Github//DeepLearningStudy//trained_model//CALTECH256(Resnet18)_Trace.pt")

## no Train Model Save


transform = torchvision.transforms.Compose([
                #torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor()
            ])

classificationDataset = torchvision.datasets.Caltech256(root="C://Github//Dataset//",
                                                        transform=transform,
                                                        download=False)

# dataset loader
data_loader = DataLoader(dataset=classificationDataset,
                         batch_size=batch_size,  # 배치 크기는 100
                         shuffle=True,
                         drop_last=True)

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
    total_batch = len(data_loader)
    print('total_batch = ', total_batch)
    for X, Y in data_loader:
        gpu_X = X.to(device)
        gpu_Y = Y.to(device)
        gpu_Y = torch.nn.functional.one_hot(gpu_Y, num_classes=257).float()

        model.train()
        optimizer.zero_grad()
        hypothesis = model(gpu_X)
        cost = criterion(hypothesis, gpu_Y)
        cost.backward()
        avg_cost += (cost / total_batch)
        optimizer.step()

        model.eval()
        hypothesis = model(gpu_X)
        correct_prediction = torch.argmax(hypothesis, 1) == torch.argmax(gpu_Y, 1)
        accuracy = correct_prediction.float().mean()
        avg_acc += (accuracy / total_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > target_accuracy:
        break;

## no Train Model Save
model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//TRAIN_CALTECH256(Resnet18).pt")
## no Train Model Save




print('Learning finished')