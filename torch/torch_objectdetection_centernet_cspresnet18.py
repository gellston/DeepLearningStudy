import torch
import torch.nn as nn
import random
import torchvision

from ptflops import get_model_complexity_info
from torchsummary import summary
from torch.utils.data import DataLoader
from util.centernet_helper import batch_loader

from model.CSPSeparableRes18CenterNet import CSPSeparableRes18CenterNet



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
batch_size = 4
target_accuracy = 0.90
learning_rate = 0.003
accuracy_threshold = 0.5
input_image_width = 512
input_image_height = 512
feature_map_scale_factor = 4
## Hyper parameter


model = CSPSeparableRes18CenterNet(class_num=257, activation=torch.nn.SiLU).to(device)
print('==== model info ====')
summary(model, (3, 512, 512))
print('====================')

macs, params = get_model_complexity_info(model,
                                         (3, input_image_height, input_image_width),
                                         as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))




## no Train Model Save
model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//FIAT(CenterNetCSPResnet18).pt")

trace_input = torch.rand(1, 3, input_image_height, input_image_width).to(device, dtype=torch.float32)
trace_model = torch.jit.trace(model, trace_input)
torch.jit.save(trace_model, "C://Github//DeepLearningStudy//trained_model//FIAT(CenterNetCSPResnet18)_Trace.pt")
## no Train Model Save



transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor()
            ])

objectDetectionDataset = torchvision.datasets.WIDERFace(root="C://Github//Dataset//",
                                                        split="val",
                                                        transform=transform,
                                                        download=False)

# dataset loader
data_loader = DataLoader(dataset=objectDetectionDataset,
                         batch_size=1,  # 배치 크기는 100
                         shuffle=True,
                         drop_last=True)

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
    total_batch = int(len(data_loader) / batch_size)

    print('total_batch = ', total_batch)


    for batch_index in range(total_batch):
        print('batch index = ', batch_index)
        image_batch, gaussian_map_batch, size_map_batch, offset_map_batch = batch_loader(data_loader,
                                                                                         batch_size,
                                                                                         input_image_width,
                                                                                         input_image_height,
                                                                                         feature_map_scale_factor)

        #print('image_batch', image_batch.shape)
        #print('gaussian_map_batch', gaussian_map_batch.shape)
        #print('size_map_batch', size_map_batch.shape)
        #print('offset_map_batch', offset_map_batch.shape)

        """
        model.train()
        optimizer.zero_grad()
        classificaiton, class_heatmap, size_map, offset_map = model(gpu_X)
        cost = criterion(classificaiton, gpu_Y)
        cost.backward()
        avg_cost += (cost / total_batch)
        optimizer.step()

        model.eval()
        classificaiton, class_heatmap, size_map, offset_map = model(gpu_X)
        correct_prediction = torch.argmax(classificaiton, 1) == torch.argmax(gpu_Y, 1)
        accuracy = correct_prediction.float().mean()
        avg_acc += (accuracy / total_batch)
        """

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > target_accuracy:
        break;

## no Train Model Save
model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//TRAIN_FIAT(CenterNetCSPResnet18).pt")
## no Train Model Save


print('Learning finished')