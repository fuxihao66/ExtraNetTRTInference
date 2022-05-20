import torch
import cv2 as cv
import config
from config import mdevice
import numpy as np
from utils import ToneSimple,DeToneSimple,ReadData,ImgReadWithPrefix,ImgRead

# img, img_2, img_3, occ_warp_img, woCheckimg, woCheckimg_2, woCheckimg_3, labelimg, metalic, Roughnessimg, Depthimg, Normalimg = ReadData(
            # "G:/TrainingSetCompressed/bunker_compressed/BK_Train_1/compressed.0168.npz")

path = "C:/Users/admin/TestSet/"
prefix = "Bunker"
idx = 157
img = ImgReadWithPrefix(path+"wrap_res", int(idx),"1", cvtrgb=True)
img_2 = ImgReadWithPrefix(path+"wrap_res", int(idx),"3", cvtrgb=True)
img_3 = ImgReadWithPrefix(path+"wrap_res", int(idx),"5", cvtrgb=True)
woCheckimg = ImgReadWithPrefix(path+"wrap_no_hole",idx,"1",prefix=prefix+config.warpPrefix,cvtrgb=True)
woCheckimg_2 = ImgReadWithPrefix(path+"wrap_no_hole",idx,"3",prefix=prefix+config.warpPrefix,cvtrgb=True)
woCheckimg_3 = ImgReadWithPrefix(path+"wrap_no_hole",idx,"5",prefix=prefix+config.warpPrefix,cvtrgb=True)
Normalimg = ImgRead(path, idx, prefix=prefix+config.TestNormalPrefix, cvtrgb=True)
metalic = ImgRead(path, idx, prefix=prefix+config.TestMetalicPrefix, cvtrgb=True)[:,:,0:1]
Depthimg = ImgRead(path, idx, prefix=prefix+config.TestDepthPrefix, cvtrgb=True)[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))
Depthimg[Depthimg > 100] = 0.0
Depthimg = (Depthimg - Depthimg.min()) / (Depthimg.max() - Depthimg.min() + 1e-6)
Roughnessimg = ImgRead(path, idx, prefix=prefix+config.TestRoughnessPrefix, cvtrgb=True)[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))
occ_warp_img = ImgReadWithPrefix(path+"occ",idx,"1",prefix=prefix+config.warpPrefix, cvtrgb=True)


input = img
mask = input.copy()
mask[mask == 0.0] = 1.0
mask[mask == -1] = 0.0
mask[mask != 0.0] = 1.0
cv.imwrite("input.exr", input)
# occ_warp_img[mask!=0.0] = 0.0
occ_warp_img[occ_warp_img < 0.0] = 0.0
woCheckimg[woCheckimg < 0.0] = 0.0
woCheckimg_2[woCheckimg_2 < 0.0] = 0.0
woCheckimg_3[woCheckimg_3 < 0.0] = 0.0

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
                        axis=2).transpose([2, 0, 1]).reshape(4, Normalimg.shape[0], Normalimg.shape[1])
his_2 = np.concatenate([woCheckimg_2, mask2[:, :, 0].reshape((Normalimg.shape[0], Normalimg.shape[1], 1))],
                        axis=2).transpose([2, 0, 1]).reshape(4, Normalimg.shape[0], Normalimg.shape[1])
his_3 = np.concatenate([woCheckimg_3, mask3[:, :, 0].reshape((Normalimg.shape[0], Normalimg.shape[1], 1))],
                        axis=2).transpose([2, 0, 1]).reshape(4, Normalimg.shape[0], Normalimg.shape[1])

hisBuffer = np.concatenate([his_3, his_2, his_1], axis=0)

# features = np.concatenate([Normalimg,Depthimg,Roughnessimg], axis=2)
features = np.concatenate([Normalimg, Depthimg, Roughnessimg, metalic], axis=2)

input = np.concatenate([occ_warp_img, features], axis=2).transpose([2, 0, 1])
input = np.concatenate([input, hisBuffer], axis=0)

with open("buffer.txt", "w") as outputFile:
    for i in range(21):
        for j in range(720):
            for k in range(1280):
                outputFile.write("{} ".format(input[i,j,k]))