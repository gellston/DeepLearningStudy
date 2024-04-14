import torch
import torchvision
import torchvision.transforms as transforms

from torchsummary import summary
from util.helper import load_infinite
from util.losses import structureLoss

from model.WideResNet import WideResNet
from model.WideResNetNormFeature import WideResNetNormFeature
from model.EfficientAD import Teacher
from model.EfficientAD import Student


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

#Hyper Parameter
batch_size=30
learning_rate = 0.001
training_epochs=200
image_width=224
image_height=224
target_accuracy=0.9
channel_size=384
#Hyper Parameter

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(200),
    transforms.ToTensor(),
])

imagenet_dataset = torchvision.datasets.ImageFolder(root='C://Dataset//ImageNet//train//',
                                                    transform=transform)

imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=0)



teacher = Teacher(with_bn=False)
teacher.load_state_dict(torch.load("C://github//DeepLearningStudy/trained_model//EfficientADTeacherBest.pth"))
teach = teacher.to(device)
teach.eval()
compiled_model = torch.jit.script(teach)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//EfficientADTeacherBest.pt")

for param in teacher.parameters():
    param.requires_grad = False

student = Student(with_bn=False).to(device)



print('==== PDN info ====')
summary(teacher, (3, 256, 256))
print('====================')

for epoch in range(training_epochs):
    avg_cost = 0
    avg_acc = 0
    current_batch = 0
    total_batch = len(imagenet_loader)
    print('total_batch = ', total_batch)
    for inputs, labels in imagenet_loader:
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            t_pdn_out = teacher(inputs)
        s_pdn_out = student(inputs)
        s_pdn_out = s_pdn_out[:, :channel_size, :, :]
        distance_s_t = torch.pow(t_pdn_out - s_pdn_out, 2)
        dhard = torch.quantile(distance_s_t, 0.999)
        hard_data = distance_s_t[distance_s_t >= dhard]
        Lhard = torch.mean(hard_data)
        #image_p = next(imagenet_iterator)
        #s_imagenet_out = student(image_p[0].cuda())
        #N = torch.mean(torch.pow(s_imagenet_out[:, :channel_size, :, :], 2))
        loss_st = Lhard #+ N


