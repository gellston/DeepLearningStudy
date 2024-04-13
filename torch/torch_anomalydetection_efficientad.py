import torch
from torchsummary import summary
from util.helper import load_infinite

from model.WideResNet import WideResNet
from model.WideResNetNormFeature import WideResNetNormFeature
from model.EfficientAD import Teacher
from model.EfficientAD import Student


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)




teacher = Teacher(with_bn=False)
teacher.load_state_dict(torch.load("C://github//DeepLearningStudy/trained_model//EfficientADTeacherBest.pth"))
teach = teacher.to(device)
teach.eval()
compiled_model = torch.jit.script(teach)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//EfficientADTeacherBest.pt")

for param in teacher.parameters():
    param.requires_grad = False

student = Student(with_bn=False)



print('==== PDN info ====')
summary(teacher, (3, 256, 256))
print('====================')

with torch.no_grad():
    t_pdn_out = teacher(image)
    normal_t_out = (t_pdn_out - self.channel_mean) / self.channel_std
s_pdn_out = student(image)
s_pdn_out = s_pdn_out[:, :self.channel_size, :, :]
distance_s_t = torch.pow(normal_t_out - s_pdn_out, 2)
dhard = torch.quantile(distance_s_t[:8, :, :, :], 0.999)
hard_data = distance_s_t[distance_s_t >= dhard]
Lhard = torch.mean(hard_data)
image_p = next(imagenet_iterator)
s_imagenet_out = student(image_p[0].cuda())
N = torch.mean(torch.pow(s_imagenet_out[:, :self.channel_size, :, :], 2))
loss_st = Lhard + N

