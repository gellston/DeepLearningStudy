import torch
import random
import torchvision
import numpy as np
import cv2
import timeit
import datetime

from torchsummary import summary
from torch.utils.data import DataLoader


from util.centernet_helper import batch_prediction_loader
from util.centernet_helper import batch_box_extractor


from model.MobileNetV3SmallCenterNet import MobileNetV3SmallCenterNet


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


## Hyper parameter
batch_size = 1
accuracy_threshold = 0.85
class_score_threshold = 0.35
iou_threshold = 0.5
input_image_width = 416
input_image_height = 416
feature_map_scale_factor = 4
pretrained = True
## Hyper parameter



#Model Setting
MobileNetV3SmallCenterNet = MobileNetV3SmallCenterNet(fpn_conv_filters=64).to(device)
print('==== model info ====')
summary(MobileNetV3SmallCenterNet, (3, input_image_height, input_image_width))
print('====================')

MobileNetV3SmallCenterNetWeight = torch.jit.load("C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(MobileNetV3SmallCenterNet).pt")
MobileNetV3SmallCenterNet.load_state_dict(MobileNetV3SmallCenterNetWeight.state_dict())

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


MobileNetV3SmallCenterNet.eval_mode()
for batch_index in range(total_batch):

    start = datetime.datetime.now()
    input_image = batch_prediction_loader(object_detection_data_loader,
                                         batch_size,
                                         input_image_width,
                                         input_image_height,
                                         device,
                                         isNorm=False)
    end = datetime.datetime.now()
    print('loading time', int((end - start).total_seconds() * 1000))


    gpu_input_image = input_image.to(device)


    start = datetime.datetime.now()
    prediction = MobileNetV3SmallCenterNet(gpu_input_image)
    end = datetime.datetime.now()
    print('inference time', int((end - start).total_seconds() * 1000))

    prediction_heatmap = prediction[0]
    prediction_features = prediction[1]
    prediction_sizemap = prediction[2]
    prediction_offsetmap = prediction[3]

    start = datetime.datetime.now()
    box_extraction = batch_box_extractor(input_image_width=input_image_width,
                                         input_image_height=input_image_height,
                                         scale_factor=feature_map_scale_factor,
                                         score_threshold=class_score_threshold,
                                         gaussian_map_batch=prediction_heatmap,
                                         size_map_batch=prediction_sizemap,
                                         offset_map_batch=prediction_offsetmap)
    end = datetime.datetime.now()
    print('box extraction time', int((end - start).total_seconds() * 1000))

    input_image = gpu_input_image[0].detach().permute(1, 2, 0).cpu().numpy()
    input_image = input_image
    input_image = input_image.astype(np.uint8)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    ##BBox Visualization
    for bbox in box_extraction[0]:
        if len(box_extraction) == 0 : break;
        bbox_x = int(bbox[0])
        bbox_y = int(bbox[1])
        bbox_width = int(bbox[2])
        bbox_height = int(bbox[3])
        cv2.rectangle(input_image, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), (0, 255, 0))
    ##BBox Visualization
    cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('input', input_image_width, input_image_height)
    cv2.imshow('input', input_image)
    cv2.waitKey(10)

