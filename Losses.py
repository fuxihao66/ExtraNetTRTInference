import torch
import numpy as np
import torch.nn as nn
from config import mLossHoleArgument,mLossHardArgument

# import pytorch_ssim
# class LossHoleAugment(nn.Module):
#     def __init__(self, para):
#         super(LossHoleAugment,self).__init__()
#         self.para = para
#     def forward(self,input,mask,target):
#         l1=torch.abs(input-target).mean()
#         lossMask=1-mask[:,:3,:,:]
#         lholeAugment=(lossMask*torch.abs(input-target)).sum()/lossMask.sum()
#         return l1 + self.para * lholeAugment


class LossHoleArgument(nn.Module):
    def __init__(self):
        super(LossHoleArgument,self).__init__()
    def forward(self,input,mask,target):
        lossMask=1-mask[:,:3,:,:]
        lholeAugment=(lossMask*torch.abs(input-target)).sum()/lossMask.sum()
        return lholeAugment
class LossHardArgument(nn.Module):
    def __init__(self,ratio=0.1):
        super(LossHardArgument,self).__init__()
        self.ratio=ratio
    def forward(self,input,target):
        n,c,h,w = input.shape
        val,ind=torch.topk(torch.abs(input-target).view(n,c,-1),k=int(h*w*self.ratio))
        return val.mean()
class mLoss(nn.Module):
    def __init__(self):
        super(mLoss,self).__init__()
        self.hole=LossHoleArgument()
        self.hard=LossHardArgument()
    def forward(self,input,mask,target):
        basicl1=torch.abs(input-target).mean()
        if mLossHoleArgument:
            basicl1+=self.hole(input,mask,target)*mLossHoleArgument
        if mLossHardArgument:
            basicl1+=self.hard(input,target)*mLossHardArgument
        return basicl1



# class SSIMLossHoleAugment(nn.Module):
#     def __init__(self, para):
#         super(SSIMLossHoleAugment,self).__init__()
#         self.para = para
#     def forward(self,input,mask,target):
#         ssim_out = -ssim_loss(input, target)
#         # l1=torch.abs(input-target).mean()
#         lossMask=1-mask[:,:3,:,:]
#         lholeAugment=(lossMask*torch.abs(input-target)).sum()/lossMask.sum()
#         return l1 + self.para * lholeAugment

'''
criterion = LossHoleAugment()
x=torch.ones(1,1,4,4)
y=torch.ones(1,1,4,4)
mask=torch.ones(1,1,4,4)
mask[:,:,1:3,1:3]=0
x[:,:,1:3,1:3]=0
loss=criterion(x,mask,y)
print(loss)
'''