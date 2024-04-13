import torch
import torch.nn.functional as F


def imagenet_norm_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to('cuda')
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to('cuda')
    x_norm = (x - mean) / (std + 1e-11)
    return x_norm


class PDN_S(torch.nn.Module):

    def __init__(self, last_kernel_size=384, with_bn=False) -> None:
        super().__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 128 3 ReLU
        # AvgPool-1 2×2 2×2 128 1 -
        # Conv-2 1×1 4×4 256 3 ReLU
        # AvgPool-2 2×2 2×2 256 1 -
        # Conv-3 1×1 3×3 256 1 ReLU
        # Conv-4 1×1 4×4 384 0 -
        self.with_bn = with_bn
        self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(256, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        return x


class Teacher(torch.nn.Module):
    def __init__(self, with_bn=False, channel_size=384, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pdn = PDN_S(last_kernel_size=channel_size, with_bn=with_bn)
    def forward(self, x):
        x = imagenet_norm_batch(x) #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        x = self.pdn(x)
        return x


class Student(torch.nn.Module):
    def __init__(self, with_bn=False, channel_size=384, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pdn = PDN_S(last_kernel_size=channel_size, with_bn=with_bn)
    def forward(self, x):
        x = imagenet_norm_batch(x) #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        x = self.pdn(x)
        return x