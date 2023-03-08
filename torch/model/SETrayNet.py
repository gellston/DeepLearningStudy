import torch

from util.helper import SESeparableActivationConv2d



class SETrayNet(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.ReLU):
        super(SETrayNet, self).__init__()

        self.class_num = class_num

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),  #256x256

            activation(),

            SESeparableActivationConv2d(in_channels=16, out_channels=32, bias=False, stride=1, #256x256
                                        activation=activation, kernel_size=3),

            SESeparableActivationConv2d(in_channels=32, out_channels=32, bias=False, stride=2, #128x128
                                        activation=activation, kernel_size=3),

            SESeparableActivationConv2d(in_channels=32, out_channels=32, bias=False, stride=1,
                                        activation=activation, kernel_size=3),

            SESeparableActivationConv2d(in_channels=32, out_channels=32, bias=False, stride=2, #64x64
                                        activation=activation, kernel_size=3),

            SESeparableActivationConv2d(in_channels=32, out_channels=32, bias=False, stride=1,
                                        activation=activation, kernel_size=3),

            SESeparableActivationConv2d(in_channels=32, out_channels=32, bias=False, stride=2, #32x32
                                        activation=activation, kernel_size=3),

            SESeparableActivationConv2d(in_channels=32, out_channels=32, bias=False, stride=1,
                                        activation=activation, kernel_size=3),

            SESeparableActivationConv2d(in_channels=32, out_channels=32, bias=False, stride=2, #16x16
                                        activation=activation, kernel_size=3),

            SESeparableActivationConv2d(in_channels=32, out_channels=32, bias=False, stride=1,
                                        activation=activation, kernel_size=3),

            SESeparableActivationConv2d(in_channels=32, out_channels=32, bias=False, stride=1,
                                        activation=activation, kernel_size=3),


            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=self.class_num,
                            kernel_size=1,
                            bias=True)
        )

        # module 초기화
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x