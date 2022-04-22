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

from model.CSPSeparableResnet18 import CSPSeparableResnet18
from model.CSPSeparableResnet18CenterNet import CSPSeparableResnet18CenterNet



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
batch_size = 15
target_accuracy = 0.90
learning_rate = 0.003
accuracy_threshold = 0.5
input_image_width = 512
input_image_height = 512
feature_map_scale_factor = 4
## Hyper parameter



separable_cspresnet18 = torch.jit.load("C://Github//DeepLearningStudy//trained_model//TRAIN_CALTECH256(CSPSeparableResnet18).pt")
CSPSeparableResnet18 = CSPSeparableResnet18(class_num=257, activation=torch.nn.SiLU).to(device)
CSPSeparableResnet18.load_state_dict(separable_cspresnet18.state_dict())


print('==== model info ====')
summary(CSPSeparableResnet18, (3, 512, 512))
print('====================')


CSPSeparableResnet18CenterNet = CSPSeparableResnet18CenterNet(backbone=CSPSeparableResnet18,
                                                              activation=torch.nn.SiLU,
                                                              pretrained=False).to(device)



## no Train Model Save
CSPSeparableResnet18CenterNet.eval()
compiled_model = torch.jit.script(CSPSeparableResnet18CenterNet)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//WIDERFACE(CSPSeparableResnet18CenterNet).pt")

trace_input = torch.rand(1, 3, input_image_height, input_image_width).to(device, dtype=torch.float32)
trace_model = torch.jit.trace(CSPSeparableResnet18CenterNet, trace_input)
torch.jit.save(trace_model, "C://Github//DeepLearningStudy//trained_model//WIDERFACE(CSPSeparableResnet18CenterNet)_Trace.pt")
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


CSPSeparableResnet18CenterNet.train()
criterion = CenterNetLoss(alpha=.25, gamma=2, lambda_size=0.1, lambda_offset=1)
optimizer = torch.optim.Adam(CSPSeparableResnet18CenterNet.parameters(), lr=learning_rate)



print("object detection training start")
for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
    total_batch = int(len(object_detection_data_loader) / batch_size)

    print('total_batch = ', total_batch)


    for batch_index in range(total_batch):
        #print('batch index = ', batch_index)
        batch = batch_loader(object_detection_data_loader,
                             batch_size,
                             input_image_width,
                             input_image_height,
                             feature_map_scale_factor,
                             device,
                             isNorm=True)
        label_image = batch[0]          #Original Image
        label_image_size = batch[1]     #Original Image Size
        label_bbox = batch[2]           #BBox Info
        label_bbox_count = batch[3]     #BBox Count
        label_heatmap = batch[4]        #Gaussian Heatmap
        label_sizemap = batch[5]        #Size Map
        label_offsetmap = batch[6]      #Offset Map

        gpu_label_image = label_image.to(device)
        gpu_label_heatmap = label_heatmap.to(device)
        gpu_label_sizemap = label_sizemap.to(device)
        gpu_label_offsetmap = label_offsetmap.to(device)


        CSPSeparableResnet18CenterNet.train()
        optimizer.zero_grad()

        prediction = CSPSeparableResnet18CenterNet(gpu_label_image)
        prediction_heatmap = prediction[0]
        prediction_features = prediction[1]
        prediction_sizemap = prediction[2]
        prediction_offsetmap = prediction[3]

        cost = criterion(prediction_features,
                         prediction_sizemap,
                         prediction_offsetmap,
                         gpu_label_heatmap,
                         gpu_label_sizemap,
                         gpu_label_offsetmap,
                         label_bbox_count,
                         batch_size)
        cost.backward()
        avg_cost += (cost / total_batch)
        optimizer.step()


        heatmap_image = prediction_heatmap[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
        cv2.namedWindow("heatmap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('heatmap', input_image_width, input_image_height)
        cv2.imshow('heatmap', heatmap_image)


        heatmap_label = label_heatmap[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
        cv2.namedWindow("heatmap_label", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('heatmap_label', input_image_width, input_image_height)
        cv2.imshow('heatmap_label', heatmap_label)



        input_image = label_image[0].detach().permute(1, 2, 0).cpu().numpy()
        input_image = input_image * 255
        input_image = input_image.astype(np.uint8)
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
    if avg_acc > accuracy_threshold:
        break;

print("object detection training end")
## no Train Model Save
trace_model.eval()
compiled_model = torch.jit.script(trace_model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPSeparableResnet18CenterNet).pt")
## no Train Model Save
print('Learning finished')