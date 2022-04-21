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




separable_cspresnet18 = torch.jit.load("C://Github//DeepLearningStudy//trained_model//CALTECH256(CSPSeparableResnet18).pt").to(device)
separable_cspresnet18.train()

print('==== backbone model info ====')
summary(separable_cspresnet18, (3, input_image_height, input_image_width))
print('====================')


