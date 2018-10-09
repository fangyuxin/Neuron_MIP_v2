from .model import Model
import torch
import torch.nn as nn

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()

        self.unet_unit = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, \
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.unet_unit(input)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()

        self.unet_up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, \
                                          kernel_size=2, stride=2)
        self.unet_unit = UnetDown(in_channels=in_channels, out_channels=out_channels)

    def forward(self, inputs1, inputs2):
        outputs1 = inputs1
        outputs2 = self.unet_up(inputs2)
        return self.unet_unit(torch.cat([outputs1, outputs2], dim=1))


class Unet(Model):
    def __init__(self, in_channels=1, out_class=2):
        super(Unet, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.unet_down1 = UnetDown(in_channels, filters[0])
        self.unet_down2 = UnetDown(filters[0], filters[1])
        self.unet_down3 = UnetDown(filters[1], filters[2])
        self.unet_down4 = UnetDown(filters[2], filters[3])
        self.center = UnetDown(filters[3], filters[4])

        self.unet_up1 = UnetUp(filters[4], filters[3])
        self.unet_up2 = UnetUp(filters[3], filters[2])
        self.unet_up3 = UnetUp(filters[2], filters[1])
        self.unet_up4 = UnetUp(filters[1], filters[0])

        self.output = nn.Conv2d(filters[0], out_class, kernel_size=1)

    def forward(self, inputs):
        x1 = self.unet_down1(inputs)
        x1_maxpool = self.maxpool(x1)
        x2 = self.unet_down2(x1_maxpool)
        x2_maxpool = self.maxpool(x2)
        x3 = self.unet_down3(x2_maxpool)
        x3_maxpool = self.maxpool(x3)
        x4 = self.unet_down4(x3_maxpool)
        x4_maxpool = self.maxpool(x4)

        center = self.center(x4_maxpool)

        up1 = self.unet_up1(x4, center)
        up2 = self.unet_up2(x3, up1)
        up3 = self.unet_up3(x2, up2)
        up4 = self.unet_up4(x1, up3)

        return self.output(up4)

