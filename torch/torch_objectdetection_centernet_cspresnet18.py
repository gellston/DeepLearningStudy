import torch
import random
import torchvision
import numpy as np
import cv2

from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR

from util.centernet_helper import batch_loader
from util.centernet_helper import batch_accuracy
from util.losses import CenterNetLoss

from model.CSPResnet18 import CSPResnet18
from model.CSPResnet18CenterNet import CSPResnet18CenterNet


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
batch_size = 22
target_accuracy = 0.90
learning_rate = 0.0001
accuracy_threshold = 0.5
input_image_width = 512
input_image_height = 512
feature_map_scale_factor = 4
pretrained = True
## Hyper parameter



#Model Setting
CSPResnet18 = CSPResnet18(class_num=1, activation=torch.nn.SiLU).to(device)
print('==== model info ====')
summary(CSPResnet18, (3, 512, 512))
print('====================')
CSPResnet18CenterNet = CSPResnet18CenterNet(backbone=CSPResnet18,
                                            activation=torch.nn.SiLU,
                                            pretrained=False).to(device)
if pretrained == True:
    CSPResnet18CenterNetBackBoneWeight = torch.jit.load("C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPResnet18CenterNetBackBone).pt")
    CSPResnet18.load_state_dict(CSPResnet18CenterNetBackBoneWeight.state_dict())
    CSPResnet18CenterNetWeight = torch.jit.load("C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPResnet18CenterNet).pt")
    CSPResnet18CenterNet.load_state_dict(CSPResnet18CenterNetWeight.state_dict())


print('==== model info ====')
summary(CSPResnet18CenterNet, (3, 512, 512))
print('====================')
#Model Setting



# object detection dataset loader
object_detection_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor()
    ])

objectDetectionDataset = torchvision.datasets.WIDERFace(root="C://Github//Dataset//",
                                                        split="train",
                                                        transform=object_detection_transform,
                                                        download=False)


object_detection_data_loader = DataLoader(dataset=objectDetectionDataset,
                                          batch_size=1,  # 배치 크기는 100
                                          shuffle=True,
                                          drop_last=True)

total_batch = int(len(object_detection_data_loader) / batch_size)
print('total batch=', total_batch)
# object detection dataset loader



CSPResnet18CenterNet.train()
criterion = CenterNetLoss(alpha=1, gamma=1, beta=0.1)
optimizer = torch.optim.SGD(CSPResnet18CenterNet.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9)
scheduler = CyclicLR(optimizer, base_lr=learning_rate, max_lr=0.05, step_size_up=30, mode='triangular2')



print("object detection training start")
for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
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


        CSPResnet18.train()
        CSPResnet18CenterNet.train()
        optimizer.zero_grad()

        prediction = CSPResnet18CenterNet(gpu_label_image)
        prediction_heatmap = prediction[0]
        prediction_features = prediction[1]
        prediction_sizemap = prediction[2]
        prediction_offsetmap = prediction[3]

        cost = criterion(prediction_features,
                         prediction_sizemap,
                         prediction_offsetmap,
                         gpu_label_heatmap,
                         gpu_label_sizemap,
                         gpu_label_offsetmap)
        cost.backward()
        avg_cost += (cost / total_batch)
        optimizer.step()

        #print('prediction_size_map count=', torch.sum(prediction_sizemap > 0).item())
        #print('prediction_offsetmap count=', torch.sum(prediction_offsetmap > 0).item())


        validation = batch_accuracy(input_image_width=input_image_width,
                                    input_image_height=input_image_height,
                                    scale_factor=feature_map_scale_factor,
                                    score_threshold=0.5,
                                    iou_threshold=0.5,
                                    gaussian_map_batch=prediction_heatmap,
                                    size_map_batch=prediction_sizemap,
                                    offset_map_batch=prediction_offsetmap,
                                    image_size_list=label_image_size,
                                    bbox_list=label_bbox)
        accuracy = validation[0]
        prediction_result = validation[1]
        avg_acc += (accuracy / total_batch)
        

        print('batch accuracy=', accuracy)

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
    #Scheduler step
    scheduler.step()

    print("학습중간에 저장")
    ## no Train Model Save
    CSPResnet18CenterNet.eval()
    CSPResnet18.eval()
    compiled_model_backbone = torch.jit.script(CSPResnet18)
    torch.jit.save(compiled_model_backbone, "C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPResnet18CenterNetBackBone).pt")
    compiled_model_head = torch.jit.script(CSPResnet18CenterNet)
    torch.jit.save(compiled_model_head, "C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPResnet18CenterNet).pt")
    ## no Train Model Save
    print('학습중간에 저장')

    print('total_batch = ', total_batch)
    print('current learning rate=', optimizer.param_groups[0]['lr'])
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > accuracy_threshold:
        break;


print("object detection training end")
## no Train Model Save
CSPResnet18CenterNet.eval()
CSPResnet18.eval()
compiled_model_backbone = torch.jit.script(CSPResnet18)
torch.jit.save(compiled_model_backbone, "C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPResnet18CenterNetBackBone).pt")
compiled_model = torch.jit.script(CSPResnet18CenterNet)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPResnet18CenterNet).pt")
## no Train Model Save
print('Learning finished')
