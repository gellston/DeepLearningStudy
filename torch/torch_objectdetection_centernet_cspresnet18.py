import torch
import random
import torchvision
import numpy as np
import cv2

from ptflops import get_model_complexity_info
from torchsummary import summary
from torch.utils.data import DataLoader

from util.centernet_helper import batch_loader
from util.losses import CenterNetLoss
from model.CSPRes18CenterNet import CSPRes18CenterNet


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


## Hyper parameter
training_epochs = 300
classification_batch_size = 5
object_detection_batch_size = 5
classification_target_accuracy = 0.90
object_detection_target_accuracy = 0.90
classification_learning_rate = 0.003
object_detection_learning_rate = 0.003
accuracy_threshold = 0.5
input_image_width = 512
input_image_height = 512
feature_map_scale_factor = 4
## Hyper parameter


model = CSPRes18CenterNet(class_num=257, activation=torch.nn.SiLU).to(device)

print('==== model info ====')
summary(model, (3, input_image_height, input_image_width))
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



# object detection dataset loader
# object detection dataset loader
object_detection_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor()
    ])

objectDetectionDataset = torchvision.datasets.WIDERFace(root="C://Github//Dataset//",
                                                        split="val",
                                                        transform=object_detection_transform,
                                                        download=False)


object_detection_data_loader = DataLoader(dataset=objectDetectionDataset,
                         batch_size=1,  # 배치 크기는 100
                         shuffle=True,
                         drop_last=True)
# object detection dataset loader
# object detection dataset loader


# classification dataset loader
# classification dataset loader
classification_transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize((input_image_width, input_image_height)),
                torchvision.transforms.ToTensor()
            ])

classificationDataset = torchvision.datasets.Caltech256(root="C://Github//Dataset//",
                                                        transform=classification_transform,
                                                        download=False)

# dataset loader
clsasification_data_loader = DataLoader(dataset=classificationDataset,
                                        batch_size=classification_batch_size,  # 배치 크기는 100
                                        shuffle=True,
                                        drop_last=True)
# classification dataset loader
# classification dataset loader



model.train()
criterion = CenterNetLoss(alpha=.25, gamma=2, lambda_size=0.1, lambda_offset=1)
crossEntropyLoss = torch.nn.CrossEntropyLoss()
classification_optimizer = torch.optim.Adam(model.parameters(), lr=classification_learning_rate)
object_detection_optimizer = torch.optim.Adam(model.parameters(), lr=object_detection_learning_rate)


print("classification training start")
##Classification Training Loop for pretraining
for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
    total_batch = len(clsasification_data_loader)

    for X, Y in clsasification_data_loader:
        gpu_X = X.to(device)
        gpu_Y = Y.to(device)
        gpu_Y = torch.nn.functional.one_hot(gpu_Y, num_classes=257).float()

        model.train()
        classification_optimizer.zero_grad()
        classification, prediction_heatmap, prediction_features, prediction_sizemap, prediction_offsetmap = model(gpu_X)
        cost = crossEntropyLoss(classification, gpu_Y)
        cost.backward()
        avg_cost += (cost / total_batch)
        classification_optimizer.step()

        model.eval()
        classification, prediction_heatmap, prediction_features, prediction_sizemap, prediction_offsetmap = model(gpu_X)
        correct_prediction = torch.argmax(classification, 1) == torch.argmax(gpu_Y, 1)
        accuracy = correct_prediction.float().mean()
        avg_acc += (accuracy / total_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > classification_target_accuracy:
        break;
print("classification training end")




print("object detection training start")
for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
    total_batch = int(len(object_detection_data_loader) / object_detection_batch_size)

    print('total_batch = ', total_batch)


    for batch_index in range(total_batch):
        #print('batch index = ', batch_index)
        label_image, label_heatmap, label_sizemap, label_offsetmap = batch_loader(object_detection_data_loader,
                                                                                  object_detection_batch_size,
                                                                                  input_image_width,
                                                                                  input_image_height,
                                                                                  feature_map_scale_factor,
                                                                                  device)
        gpu_label_image = label_image.to(device)
        gpu_label_heatmap = label_heatmap.to(device)
        gpu_label_sizemap = label_sizemap.to(device)
        gpu_label_offsetmap = label_offsetmap.to(device)


        model.train()
        object_detection_optimizer.zero_grad()
        classification, prediction_heatmap, prediction_features, prediction_sizemap, prediction_offsetmap = model(gpu_label_image)

        cost = criterion(prediction_features,
                         prediction_sizemap,
                         prediction_offsetmap,
                         gpu_label_heatmap,
                         gpu_label_sizemap,
                         gpu_label_offsetmap)

        cost.backward()
        avg_cost += (cost / total_batch)
        object_detection_optimizer.step()


        heatmap_image = prediction_heatmap[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
        cv2.namedWindow("heatmap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('heatmap', input_image_width, input_image_height)
        cv2.imshow('heatmap', heatmap_image)


        heatmap_label = label_heatmap[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
        cv2.namedWindow("heatmap_label", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('heatmap_label', input_image_width, input_image_height)
        cv2.imshow('heatmap_label', heatmap_label)



        input_image = label_image[0].detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("input", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('input', input_image_width, input_image_height)
        cv2.imshow('input', input_image)
        cv2.waitKey(10)


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
    if avg_acc > object_detection_target_accuracy:
        break;

print("object detection training end")

## no Train Model Save
model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//TRAIN_FIAT(CenterNetCSPResnet18).pt")
## no Train Model Save


print('Learning finished')