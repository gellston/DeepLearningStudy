import torch
import random
import torchvision
import numpy as np
import cv2
import gc

from torchsummary import summary
from torch.utils.data import DataLoader

from util.centernet_helper import batch_loader
from util.centernet_helper import batch_accuracy
from util.losses import CenterNetLossV2

from model.CSPMobileNetV2 import CSPMobileNetV2
from model.CSPMobileNetV2CenterNet import CSPMobileNetV2CenterNet


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)





## Hyper parameter
training_epochs = 140
current_epoch = 41
batch_size = 7
learning_rate = 0.0005
accuracy_threshold = 0.80
class_score_threshold = 0.5
iou_threshold = 0.5
input_image_width = 640
input_image_height = 640
feature_map_scale_factor = 4
pretrained_centernet = True
pretrained_backbone = True
validation_check = False
training_check = False
## Hyper parameter



#Model Setting
CSPMobileNetV2 = CSPMobileNetV2(class_num=1, activation=torch.nn.ReLU6).to(device)
print('==== model info ====')
summary(CSPMobileNetV2, (3, 640, 640))
print('====================')
CSPMobileNetV2CenterNet = CSPMobileNetV2CenterNet(backbone=CSPMobileNetV2,
                                                  pretrained=False).to(device)

if pretrained_backbone == True:
    CSPMobileNetV2CenterNetBackBoneWeight = torch.jit.load("C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPMobileNetV2CenterNetBackBone).pt")
    CSPMobileNetV2.load_state_dict(CSPMobileNetV2CenterNetBackBoneWeight.state_dict())

if pretrained_centernet == True:
    CSPMobileNetV2CenterNetWeight = torch.jit.load("C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPMobileNetV2CenterNet).pt")
    CSPMobileNetV2CenterNet.load_state_dict(CSPMobileNetV2CenterNetWeight.state_dict())


print('==== model info ====')
summary(CSPMobileNetV2CenterNet, (3, 640, 640))
print('====================')
#Model Setting



# object detection dataset loader
object_detection_transform = torchvision.transforms.Compose([
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


CSPMobileNetV2CenterNet.train()
criterion = CenterNetLossV2(alpha=1, gamma=1, beta=0.1)
optimizer = torch.optim.RAdam(CSPMobileNetV2CenterNet.parameters(), lr=learning_rate)

for epoch in range(current_epoch, training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0

    if epoch > 90 and epoch < 120:
        learning_rate = learning_rate / 10
        for g in optimizer.param_groups:
            g['lr'] = learning_rate

    for batch_index in range(total_batch):
        batch = batch_loader(object_detection_data_loader,
                             batch_size,
                             input_image_width,
                             input_image_height,
                             feature_map_scale_factor,
                             device,
                             isNorm=False)
        label_image = batch[0]          #Original Image
        label_bbox = batch[1]           #BBox Info
        label_heatmap = batch[2]        #Gaussian Heatmap
        label_sizemap = batch[3]        #Size Map
        label_offsetmap = batch[4]      #Offset Map

        gpu_label_image = label_image.to(device)
        gpu_label_heatmap = label_heatmap.to(device)
        gpu_label_sizemap = label_sizemap.to(device)
        gpu_label_offsetmap = label_offsetmap.to(device)


        CSPMobileNetV2.train()
        CSPMobileNetV2CenterNet.train()
        optimizer.zero_grad()

        prediction = CSPMobileNetV2CenterNet(gpu_label_image)
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


        if validation_check == True:
            CSPMobileNetV2.eval()
            CSPMobileNetV2CenterNet.eval()
            validation = batch_accuracy(input_image_width=input_image_width,
                                        input_image_height=input_image_height,
                                        scale_factor=feature_map_scale_factor,
                                        score_threshold=class_score_threshold,
                                        iou_threshold=iou_threshold,
                                        gaussian_map_batch=prediction_heatmap,
                                        size_map_batch=prediction_sizemap,
                                        offset_map_batch=prediction_offsetmap,
                                        bbox_list=label_bbox)
            accuracy = validation[0]
            prediction_result = validation[1]
            avg_acc += (accuracy / total_batch)
            print('batch accuracy=', accuracy)


        if training_check == True:
            heatmap_image = prediction_heatmap[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
            cv2.namedWindow("heatmap", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('heatmap', input_image_width, input_image_height)
            cv2.imshow('heatmap', heatmap_image)


            heatmap_label = label_heatmap[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32)
            cv2.namedWindow("heatmap_label", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('heatmap_label', input_image_width, input_image_height)
            cv2.imshow('heatmap_label', heatmap_label)



            input_image = label_image[0].detach().permute(1, 2, 0).cpu().numpy()
            input_image = input_image
            input_image = input_image.astype(np.uint8)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
            ##BBox Visualization
            for bbox in label_bbox[0]:
                bbox_x = int(bbox[0])
                bbox_y = int(bbox[1])
                bbox_width = int(bbox[2])
                bbox_height = int(bbox[3])
                cv2.rectangle(input_image, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), (255, 0, 0))
            ##BBox Visualization
            cv2.namedWindow("input", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('input', input_image_width, input_image_height)
            cv2.imshow('input', input_image)
            cv2.waitKey(10)

        gc.collect()

    print("학습중간에 저장")
    ## no Train Model Save
    CSPMobileNetV2CenterNet.eval()
    CSPMobileNetV2.eval()
    compiled_model_backbone = torch.jit.script(CSPMobileNetV2)
    torch.jit.save(compiled_model_backbone, "C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPMobileNetV2CenterNetBackBone).pt")
    compiled_model_head = torch.jit.script(CSPMobileNetV2CenterNet)
    torch.jit.save(compiled_model_head, "C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPMobileNetV2CenterNet).pt")
    ## no Train Model Save
    print('학습중간에 저장')

    print('total_batch = ', total_batch)
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > accuracy_threshold:
        break;


## no Train Model Save
CSPMobileNetV2CenterNet.eval()
CSPMobileNetV2.eval()
compiled_model_backbone = torch.jit.script(CSPMobileNetV2)
torch.jit.save(compiled_model_backbone, "C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPMobileNetV2CenterNetBackBone).pt")
compiled_model = torch.jit.script(CSPMobileNetV2CenterNet)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(CSPMobileNetV2CenterNet).pt")
## no Train Model Save

print('Learning finished')
