import gc
import torch
import torch.nn as nn
import random

from torchsummary import summary
from torch.utils.data import DataLoader
from util.helper import IOU

from util.TorchSegmentationDatasetLoaderV1 import TorchSegmentationDatasetLoaderV1
from model.CustomSegmentationV1 import CustomSegmentationV1


gc.collect()
torch.cuda.set_per_process_memory_fraction(1.0)
torch.cuda.empty_cache()


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


training_epochs = 300
batch_size = 5

target_accuracy = 0.99


datasets = TorchSegmentationDatasetLoaderV1('C://Github//DeepLearningStudy//dataset//portrait_segmentation_input256x256',
                                            'C://Github//DeepLearningStudy//dataset//portrait_segmentation_label256x256',
                                            image_height=256,
                                            image_width=256,
                                            isColor=True,
                                            isNorm=False)

data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)



CustomSegmentationV1 = CustomSegmentationV1().to(device)
print('==== model info ====')
summary(CustomSegmentationV1, (3, 256, 256))
print('====================')


loss_fn = nn.BCELoss().to(device)# 내부적으로 소프트맥스 함수를 포함하고 있음.

optimizer = torch.optim.Adam(CustomSegmentationV1.parameters(), lr=0.003)



total_batch = len(data_loader)
print('total_batch=', total_batch)

final_cost = 0
final_accuracy = 0

for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0


    for X, Y in data_loader:
        gpu_X = X.to(device)
        gpu_Y = Y.to(device)

        ##cost calculation
        CustomSegmentationV1.train()
        optimizer.zero_grad()
        hypothesis = CustomSegmentationV1(gpu_X)
        cost = loss_fn(hypothesis, gpu_Y)
        cost.backward()
        optimizer.step()

        ##acc calculation
        CustomSegmentationV1.eval()
        prediction = CustomSegmentationV1(gpu_X)
        accuracy = IOU(gpu_Y.cpu().detach().numpy(), prediction.cpu().detach().numpy())

        del gpu_X
        del gpu_Y
        torch.cuda.empty_cache()

        avg_cost += (cost / total_batch)
        avg_acc += (accuracy / total_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > target_accuracy:
        final_accuracy = avg_acc
        final_cost = avg_cost
        break;

## no Train Model Save
CustomSegmentationV1.eval()
trace_input = torch.rand(1, 3, 256, 256).to(device, dtype=torch.float32)
traced_script_module = torch.jit.trace(CustomSegmentationV1, trace_input)
traced_script_module.save("C://Github//DeepLearningStudy//trained_model//CustomSegmentationV1.pt")
## no Train Model Save


print('Final accuracy = ', final_accuracy, ', cost = ', final_cost)
print('Learning finished')




