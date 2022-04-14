import torch

class VGG16BN_GAP(torch.nn.Module):

    def __init__(self, class_num=5):
        super(VGG16BN_GAP, self).__init__()
        self.drop_rate = 0.3
        self.class_num = class_num

        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(64),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(64),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(128),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(128),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(256),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(256),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(256),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(512),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(512),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(512),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(512),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(512),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False),
                                          torch.nn.BatchNorm2d(512),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.final_conv_layer = torch.nn.Sequential(torch.nn.Conv2d(512, self.class_num, kernel_size=3, stride=1, padding='same', bias=True))


        self.global_average_pooling = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.final_conv_layer(x)
        x = self.global_average_pooling(x)
        x = x.view(-1, self.class_num)
        x = self.sigmoid(x)

        return x