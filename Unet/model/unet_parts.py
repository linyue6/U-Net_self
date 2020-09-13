''' Parts of the U-Net model '''
import torch
import torch.nn as nn
import torch.nn.functional as F

'''修改网络，使网络的输出尺寸正好等于图片的输入尺寸'''
class DoubleConv(nn.Module):
    # (convolution => [BN] => ReLU) * 2
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            # 图片卷积处理，使图片中有差异的部分从视觉上扩大差异
            # in_channel: 输入数据的通道数
            # out_channel: 输出数据的通道数
            # kennel_size: 卷积核大小
            # padding: 填充数
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 数据归一化处理，防止一些数据由于数值过大而使得数值很小的数据失去作用，而被忽略
            nn.BatchNorm2d(out_channels),
            # 计算激活函数ReLU,即max(feature, 0),将每一行非最大值置0
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    ''' Downscaling with maxpool then double conv '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forword(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    ''' Upscaling then double conv '''

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forword(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # //, 地板除
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim = 1)
        print(x1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)
