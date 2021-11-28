import torch

class VGG16FC(torch.nn.Module):

    def __init__(self, class_num=5):
        super(VGG16FC, self).__init__()
        self.drop_rate = 0.3
        self.class_num = class_num

        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.adaptive_average_pool = torch.nn.AdaptiveAvgPool2d((7, 7))

        # L1 FC 7x7x512 inputs ->
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(7*7*512, 4096, bias=True),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=self.drop_rate))

        self.fc2 = torch.nn.Sequential(torch.nn.Linear(4096, 4096, bias=True),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=self.drop_rate))

        self.final_output = torch.nn.Sequential(torch.nn.Linear(4096, self.class_num, bias=True),
                                                torch.nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.adaptive_average_pool(x)
        x = x.view(-1, 7*7*512)
        x = self.fc1(x)
        x = x.view(-1, 4096)
        x = self.fc2(x)
        x = self.final_output(x)

        return x