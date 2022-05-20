#----------------------------------这里用的是原版gated


import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.models.resnet import BasicBlock
import torch
import numpy as np
import torch.nn as nn
from config import mdevice
# import softsplat


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None ,kernel_size=3,padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, kernel_size=kernel_size,
                               padding=padding)
    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)
class LWGatedConv2D(nn.Module):
    def __init__(self, input_channel1, output_channel, pad, kernel_size, stride):
        super(LWGatedConv2D, self).__init__()

        self.conv_feature = nn.Conv2d(in_channels=input_channel1, out_channels=output_channel, kernel_size=kernel_size,
                                      stride=stride, padding=pad)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(in_channels=input_channel1, out_channels=1, kernel_size=kernel_size, stride=stride,
                      padding=pad),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # inputs = inputs * mask
        newinputs = self.conv_feature(inputs)
        mask = self.conv_mask(inputs)

        return newinputs*mask
class DownLWGated(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = LWGatedConv2D(in_channels, in_channels, kernel_size=3, pad=1, stride=2)
        self.conv1 = LWGatedConv2D(in_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = LWGatedConv2D(out_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x= self.downsample(x)
        x= self.conv1(x)
        x = self.relu1(self.bn1(x))
        x= self.conv2(x)
        x = self.relu2(self.bn2(x))
        return x

class FlowPred(nn.Module):
    def __init__(self,in_channels,num_his=3):
        super(FlowPred, self).__init__()
        self.down1=nn.Sequential(nn.Conv2d(in_channels,in_channels//num_his,kernel_size=1),
                                 nn.BatchNorm2d(in_channels//num_his))
        self.block1=BasicBlock(in_channels,in_channels//num_his,downsample=self.down1)
        self.block2=BasicBlock(in_channels//num_his,in_channels//num_his)
        self.outc=nn.Conv2d(in_channels//num_his,2,kernel_size=1)
    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        return self.outc(x)


def torchImgWrap(img1,motion2):
    n,c,h,w=img1.shape
    dx,dy=torch.linspace(-1,1,w).to(img1.device),torch.linspace(-1,1,h).to(img1.device)
    grid_y, grid_x = torch.meshgrid(dy, dx)
    grid_x = grid_x.repeat(n,1,1)-(2*motion2[:,0]/(w))
    grid_y = grid_y.repeat(n,1,1)+(2*motion2[:,1]/(h))
    cood = torch.stack([grid_x, grid_y], dim=-1)
    res=F.grid_sample(img1, cood, align_corners=True)
    return res


backwarp_tenGrid = {}
def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(mdevice)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=torch.clamp(g, -1, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5824fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class UNetLWGated(nn.Module):
    # a tinyer Unet which only has 3 downsample pass
    def __init__(self, n_channels, n_classes, bilinear=False, skip=True):
        super(UNetLWGated, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.skip = skip

        self.convHis1 = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.convHis2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.convHis3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # self.flowpred=FlowPred(64*3)
        self.lowlevelGated = LWGatedConv2D(32*3, 32, kernel_size=3, stride=1, pad=1)

        self.conv1 = LWGatedConv2D(n_channels, 24, kernel_size=3, stride=1, pad=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = LWGatedConv2D(24, 24, kernel_size=3, stride=1, pad=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.relu2 = nn.ReLU(inplace=True)
        self.down1 = DownLWGated(24, 24)
        self.down2 = DownLWGated(24, 32)
        self.down3 = DownLWGated(32, 32)
        # self.up1 = ConvUp(64, 64)
        # self.up2 = ConvUp(64, 24)
        # self.up3 = ConvUp(24, 24)
        self.up1 = Up(96, 32)
        self.up2 = Up(56, 24)
        self.up3 = Up(48, 24)
        self.outc = nn.Conv2d(24, n_classes, kernel_size=1)
        # self.sig = nn.Sigmoid()
    def forward(self, warp,warp_occ, normal,hisBuffer1, hisBuffer2, hisBuffer3):
        # x, feature, mask, hisBuffer
        # hisBuffer = hisBuffer.reshape(-1, 4, hisBuffer.shape[-2], hisBuffer.shape[-1])
        warp = warp.permute(0,3,2,1)
        warp_occ = warp_occ.permute(0,3,2,1)
        normal = normal.permute(0,3,2,1)
        hisBuffer1 = hisBuffer1.permute(0,3,2,1)
        hisBuffer2 = hisBuffer2.permute(0,3,2,1)
        hisBuffer3 = hisBuffer3.permute(0,3,2,1)
        mask = hisBuffer1[:,0:1,:,:]

        height = warp.shape[2]
        width = warp.shape[3]


        hisBuffer = torch.cat([hisBuffer1, hisBuffer2, hisBuffer3], dim=0)
        # hisBuffer[:,0:3,:,:] = torch.log(hisBuffer[:,0:3,:,:] + 1.0)


        hisDown1 = self.convHis1(hisBuffer)
        hisDown2 = self.convHis2(hisDown1)
        hisDown3 = self.convHis3(hisDown2)
        cathisDown3 = hisDown3.reshape(-1, 3*32, hisDown3.shape[-2], hisDown3.shape[-1])  # 64

        #cathisDown3=torch.cat([hisDown3,hisfeature1_3,hisfeature2_3],dim=1)
        motionFeature = self.lowlevelGated(cathisDown3)

        x = torch.cat([warp[:,0:3,:,:], warp_occ[:,0:3,:,:]], dim=1) 
        # x = torch.log(x + 1.0)

        feature = torch.cat([normal, warp[:,3:4,:,:], warp_occ[:,3:4,:,:]], dim=1)
        x1=torch.cat([x,x*mask, feature],dim=1)
        x1= self.conv1(x1)
        x1 = self.relu1(self.bn1(x1))
        x1= self.conv2(x1)
        x1 = self.relu2(self.bn2(x1))

        x2= self.down1(x1)
        x3= self.down2(x2)
        x4= self.down3(x3)

        # x4=warp(x4,flow3)
        x4 = torch.cat([x4, motionFeature], dim=1)
        # x4 = softsplat.FunctionSoftsplat(tenInput=x4, tenFlow=flow3, tenMetric=None, strType='average')
        res = self.up1(x4, x3)
        res= self.up2(res, x2)
        res= self.up3(res, x1)
        logits = self.outc(res)
        if self.skip:
            logits = logits + x[:, 0:3, :, :]

        logits = torch.cat([logits, torch.zeros((1, 1, height, width)).to(mdevice)], dim=1).permute(0, 3,2,1)
        
        return logits#,flow3,hisDown3