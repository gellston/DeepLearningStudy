import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

from torchsummary import summary
from util.helper import load_infinite
from util.helper import teacher_normalization
from util.MVTecAnomalyDataset import MVTecAnomalyDataset

from model.WideResNet import WideResNet
from model.WideResNetNormFeature import WideResNetNormFeature
from model.EfficientAD import Teacher
from model.EfficientAD import Student


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

#Hyper Parameter
batch_size=8
learning_rate = 0.0001
weight_decay = 0.00001
training_epochs=200
image_width=256
image_height=256
target_accuracy=0.9
channel_size=384
best_loss = 0.1
#Hyper Parameter

transform = transforms.Compose([
    transforms.Resize(image_width),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(200),
    transforms.ToTensor(),
])

imagenet_dataset = torchvision.datasets.ImageFolder(root='E://데이터셋//ImageNet//val//',
                                                    transform=transform)

imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
infinite_imagenet_loader = load_infinite(imagenet_loader)

mvtech_dataset = MVTecAnomalyDataset(imagePath="E://데이터셋//mvtec_anomaly_detection//bottle//train//good//",
                                     image_height=image_height,
                                     image_width=image_width,
                                     isColor=True)

mvtech_loader = torch.utils.data.DataLoader(mvtech_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


mvtech_eval_dataset = MVTecAnomalyDataset(imagePath="E://데이터셋//mvtec_anomaly_detection//bottle//test//broken_large//",
                                          image_height=image_height,
                                          image_width=image_width,
                                          isColor=True)

mvtech_eval_loader = torch.utils.data.DataLoader(mvtech_eval_dataset, batch_size=1, shuffle=False, num_workers=0)


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

teacher_mean, teacher_std = teacher_normalization(teacher, mvtech_loader)

optimizer = torch.optim.Adam(student.parameters(),lr=learning_rate,weight_decay=weight_decay)

for epoch in range(training_epochs):
    avg_loss = 0
    current_batch = 0
    total_batch = len(mvtech_loader)
    for inputs in mvtech_loader:
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        with torch.no_grad():
            t_pdn_out = teacher(inputs)
            t_pdn_out = (t_pdn_out - teacher_mean) / teacher_std
        s_pdn_out = student(inputs)
        s_pdn_out = s_pdn_out[:, :channel_size, :, :]
        distance_s_t = torch.pow(t_pdn_out - s_pdn_out, 2)
        dhard = torch.quantile(distance_s_t[:8, :, :, :], 0.999)
        hard_data = distance_s_t[distance_s_t >= dhard]
        Lhard = torch.mean(hard_data)
        #imagenet iteration
        image_p = next(infinite_imagenet_loader)
        s_imagenet_out = student(image_p[0].cuda())
        N = torch.mean(torch.pow(s_imagenet_out[:, :channel_size, :, :], 2))

        loss_st = Lhard + N
        #backpropagation
        optimizer.zero_grad()
        loss_st.backward()
        optimizer.step()

        avg_loss += (loss_st.item() / total_batch)
        
    print('current avg loss = ', avg_loss)
    if avg_loss < best_loss:
        break
print("anomaly detection model training end.")
student.eval()
compiled_model = torch.jit.script(student)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//EfficientADStudentBest.pt")

def predict(image, teacher, student, teacher_mean, teacher_std):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    map_st = torch.mean((teacher_output - student_output)**2, dim=1, keepdim=True)
    return map_st

for inputs in mvtech_eval_loader:
    inputs = inputs.to(device)
    anomal_map = predict(inputs, teacher, student, teacher_mean, teacher_std)
    input_image = anomal_map[0][0].detach().cpu().numpy()
    input_image = input_image
    input_image = cv2.normalize(input_image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('input', image_width, image_height)
    cv2.imshow('input', input_image)
    cv2.waitKey()



