import torch
from torch2trt import torch2trt
from Small_UnetGated import UNetLWGated,ConvHis
import numpy as np
from utils import ToneSimple,DeToneSimple,ReadData
import cv2 as cv
def Tensor2NP(t):
    res=t.squeeze(0).cpu().numpy().transpose([1,2,0])
    res=cv.cvtColor(res,cv.COLOR_RGB2BGR)
    res = DeToneSimple(res)

    return res
def doForward():
    with torch.no_grad():
        img, img_2, img_3, occ_warp_img, woCheckimg, woCheckimg_2, woCheckimg_3, labelimg, metalic, Roughnessimg, Depthimg, Normalimg = ReadData("compressed.0145.npy")

        input = img
        mask = input.copy()
        mask[mask == 0.0] = 1.0
        mask[mask == -1] = 0.0
        mask[mask != 0.0] = 1.0

        # occ_warp_img[mask!=0.0] = 0.0
        occ_warp_img[occ_warp_img < 0.0] = 0.0
        woCheckimg[woCheckimg < 0.0] = 0.0
        woCheckimg_2[woCheckimg_2 < 0.0] = 0.0
        woCheckimg_3[woCheckimg_3 < 0.0] = 0.0

        labelimg = ToneSimple(labelimg)
        occ_warp_img = ToneSimple(occ_warp_img)
        woCheckimg = ToneSimple(woCheckimg)

        mask2 = img_2.copy()
        mask2[mask2 == 0.] = 1.0
        mask2[mask2 == -1] = 0.0
        mask2[mask2 != 0.0] = 1.0

        mask3 = img_3.copy()
        mask3[mask3 == 0.] = 1.0
        mask3[mask3 == -1] = 0.0
        mask3[mask3 != 0.0] = 1.0

        # occ_warp_img_2[mask2!=0.0] = 0.0
        # occ_warp_img_2[occ_warp_img_2 < 0.0] = 0.0
        # occ_warp_img_2 = ToneSimple(occ_warp_img_2)
        woCheckimg_2 = ToneSimple(woCheckimg_2)

        # occ_warp_img_3[mask3!=0.0] = 0.0
        # occ_warp_img_3[occ_warp_img_3 < 0.0] = 0.0
        # occ_warp_img_3 = ToneSimple(occ_warp_img_3)
        woCheckimg_3 = ToneSimple(woCheckimg_3)

        his_1 = np.concatenate([woCheckimg, mask[:, :, 0].reshape((Normalimg.shape[0], Normalimg.shape[1], 1))],
                               axis=2).transpose([2, 0, 1]).reshape(1, 4, Normalimg.shape[0], Normalimg.shape[1])
        his_2 = np.concatenate([woCheckimg_2, mask2[:, :, 0].reshape((Normalimg.shape[0], Normalimg.shape[1], 1))],
                               axis=2).transpose([2, 0, 1]).reshape(1, 4, Normalimg.shape[0], Normalimg.shape[1])
        his_3 = np.concatenate([woCheckimg_3, mask3[:, :, 0].reshape((Normalimg.shape[0], Normalimg.shape[1], 1))],
                               axis=2).transpose([2, 0, 1]).reshape(1, 4, Normalimg.shape[0], Normalimg.shape[1])

        hisBuffer = np.concatenate([his_3, his_2, his_1], axis=0)

        # features = np.concatenate([Normalimg,Depthimg,Roughnessimg], axis=2)
        features = np.concatenate([Normalimg, Depthimg, Roughnessimg, metalic], axis=2)

        input = np.concatenate([woCheckimg, occ_warp_img], axis=2).transpose([2, 0, 1])
        labelimg = labelimg.transpose([2, 0, 1])
        features = features.transpose([2, 0, 1])
        finalMask = np.repeat(mask[:, :, 0].reshape((Normalimg.shape[0], Normalimg.shape[1], 1)), 6, axis=2).transpose(
            [2, 0, 1])

        hisBuffer = torch.tensor(hisBuffer).cuda()
        input = torch.tensor(input).unsqueeze(0).cuda()
        features = torch.tensor(features).unsqueeze(0).cuda()
        finalMask = torch.tensor(finalMask).unsqueeze(0).cuda()
        labelimg = torch.tensor(labelimg).unsqueeze(0).cuda()

        model_unet = UNetLWGated(18, 3)
        model_unet.load_state_dict(torch.load("totalModel.140.pth.tar")["state_dict"], strict=False)
        model_unet = model_unet.cuda().eval()

        model_his = ConvHis()
        model_his.load_state_dict(torch.load("totalModel.140.pth.tar")["state_dict"], strict=False)
        model_his = model_his.cuda().eval()


        model_his_trt = torch2trt(model_his, [hisBuffer], max_batch_size=3)
        #model_unet_trt = torch2trt(model_unet, [x, feature, mask, cathisDown3])

        his_y = model_his(hisBuffer)
        his_y_trt = model_his_trt(hisBuffer)

        his_y = his_y.reshape(1, 192, 90, 160)
        his_y_trt = his_y_trt.reshape(1, 192, 90, 160)

        model_unet_trt = torch2trt(model_unet, [input, features, finalMask, his_y])

        y = model_unet(input, features, finalMask, his_y)
        y_trt = model_unet_trt(input, features, finalMask, his_y_trt)
        print(torch.mean(torch.abs(y - y_trt)))


        '''
        his_y = model_his(hisBuffer)
        his_y = his_y.reshape(1, 192, 90, 160)
        res = model_unet(input, features, finalMask, his_y)
        '''

        cv.imwrite("./resTrain/res train %s.exr"%("trt"), Tensor2NP(y_trt))
        cv.imwrite("./resTrain/res train %s.exr" % ("gt"), Tensor2NP(y))


doForward()
'''
x=torch.randn(1,6,720,1280).cuda()
feature=torch.randn(1,6,720,1280).cuda()
mask=torch.randn(1,6,720,1280).cuda()
hisBuffer=torch.randn(3,4,720,1280).cuda()
cathisDown3 = torch.randn(1,192,90,160).cuda()
model_unet = UNetLWGated(18, 3)
model_unet.load_state_dict(torch.load("totalModel.140.pth.tar")["state_dict"],strict=False)
model_unet = model_unet.cuda().eval()


model_his = ConvHis()
model_his.load_state_dict(torch.load("totalModel.140.pth.tar")["state_dict"],strict=False)
model_his=model_his.cuda().eval()

input_names_his = [ "input_his"]
output_names_his = [ "output_his"]

input_names_unet = [ "input_x","input_feature","input_mask","input_cathis"]
output_names_unet = [ "output1"]


model_his_trt = torch2trt(model_his,[hisBuffer],max_batch_size=3)
model_unet_trt = torch2trt(model_unet, [x,feature,mask,cathisDown3])

his_y = model_his(hisBuffer)
his_y_trt = model_his_trt(hisBuffer)

his_y=his_y.reshape(1,192,90,160)
his_y_trt=his_y_trt.reshape(1,192,90,160)

y=model_unet(x,feature,mask,his_y)
y_trt=model_unet_trt(x,feature,mask,his_y_trt)

# check the output against PyTorch
print(torch.mean(torch.abs(y - y_trt)))
doForward(model_his,model_unet,"gt")
doForward(model_his_trt,model_unet_trt,"trt")
'''
