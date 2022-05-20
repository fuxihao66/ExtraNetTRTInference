# from backup.Small_UnetGated import UNetLWGated_lessc
from Small_UnetGated import UNetLWGated
import torch
from config import mdevice

width = 1280
height = 720
# width = 720
# height = 480

# warp=torch.randn(1,4,height,width).to(mdevice)
# warp_occ=torch.randn(1,4,height,width).to(mdevice)
# normal=torch.randn(1,4,height,width).to(mdevice)
# mask=torch.randn(1,1,height,width).to(mdevice)
# hisBuffer1=torch.randn(1,4,height,width).to(mdevice)
# hisBuffer2=torch.randn(1,4,height,width).to(mdevice)
# hisBuffer3=torch.randn(1,4,height,width).to(mdevice)


warp=torch.randn(1, width, height, 4).to(mdevice)
warp_occ=torch.randn(1, width, height, 4).to(mdevice)
normal=torch.randn(1, width, height, 4).to(mdevice)
# mask=torch.randn(1, width, height, 1).to(mdevice)
hisBuffer1=torch.randn(1, width, height, 4).to(mdevice)
hisBuffer2=torch.randn(1, width, height, 4).to(mdevice)
hisBuffer3=torch.randn(1, width, height, 4).to(mdevice)

model = UNetLWGated(18, 3)
model.load_state_dict(torch.load("totalModel.290.pth.tar")["state_dict"])
model = model.to(mdevice)
model.eval()
input_names = [ "warp_no_hole", "warp_occ","normal", "mask_one_channel", "history_1","history_2", "history_3"]
output_names = [ "output_1"]
print(model.conv1.conv_feature.weight[0,0])

torch.onnx.export(model, (warp,warp_occ, normal,hisBuffer1, hisBuffer2, hisBuffer3),
                  "UNetGated.onnx", verbose=True, input_names=input_names, output_names=output_names,opset_version=11)