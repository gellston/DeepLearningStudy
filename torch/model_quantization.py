import torch
import torchvision
import numpy as np
import cv2
import gc

from torchsummary import summary
from torch.utils.data import DataLoader

from util.centernet_helper import batch_loader
from util.centernet_helper import batch_accuracy
from util.losses import CenterNetLossV2

from model.MobileNetV3SmallCenterNet import MobileNetV3SmallCenterNet



device = "cpu" # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


model = MobileNetV3SmallCenterNet(fpn_conv_filters=64).to(device)
MobileNetV3CenterNetWeight = torch.jit.load("C://Github//DeepLearningStudy//trained_model//TRAIN_WIDERFACE(MobileNetV3SmallCenterNet).pt")
model.load_state_dict(MobileNetV3CenterNetWeight.state_dict())

quantized_model = torch.quantization.quantize_dynamic(model,
                                                      {
                                                          torch.nn.Conv2d,
                                                          torch.nn.Linear,
                                                          torch.nn.ReLU6,
                                                          torch.nn.ReLU,
                                                          torch.nn.BatchNorm2d,
                                                          torch.nn.SiLU,
                                                          torch.nn.Hardswish
                                                      }
                                                      , dtype=torch.qint8)

## no Train Model Save
quantized_model.eval_mode()
compiled_model = torch.jit.script(quantized_model)
torch.jit.save(compiled_model,
               "C://Github//DeepLearningStudy//trained_model//Quantization(MobileNetV3SmallCenterNet).pt")
## no Train Model Save