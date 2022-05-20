


import torch
from torchvision import models
from torchprofile import profile_macs
from Small_UnetGated import UNetLWGated

import os

inputs = torch.randn(1, 21, 800, 1560)

# ls=dir(models)
# ls.sort()
# res=[]
# for i in ls:
#     if not (i.startswith('_')) and (i!='utils.py'):
#         res.append(i.split('.')[0])
# print(res)
# macs_results={}
# for i in res:
#     try:
        
#         macs_results[i]=MACs
#     except Exception as e:
#         pass
model = UNetLWGated(18, 3)
MACs=profile_macs(model,inputs)
print(MACs)