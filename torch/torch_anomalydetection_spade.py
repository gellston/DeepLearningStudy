import torchvision.models as models


resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in resnet18.parameters():
    param.requires_grad = False
resnet18.eval()

print('test')
