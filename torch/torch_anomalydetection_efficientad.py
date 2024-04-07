import torch

from torchsummary import summary
from model.WideResNet import WideResNet
from model.WideResNetNormFeature import WideResNetNormFeature
from model.PDN import PDN_S

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

wideresnet = WideResNet(class_num=4).to(device)
print('==== wide resnet info ====')
summary(wideresnet, (3, 256, 256))
print('====================')


wideresnet_feature = WideResNetNormFeature(wide_resnet=wideresnet)
print('==== wide resnet feature info ====')
summary(wideresnet_feature, (3, 256, 256))
print('====================')

trace_input = torch.rand(1, 3, 256, 256).to(device, dtype=torch.float32)
features = wideresnet_feature(trace_input)


PDN_S = PDN_S().to(device)
print('==== PDN info ====')
summary(PDN_S, (3, 256, 256))
print('====================')


