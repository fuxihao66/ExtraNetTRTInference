import torch.utils.data as data
import torch
import os
import cv2 as cv
import numpy as np
import config
import numpy as np 
from matplotlib import pyplot as plt 
 


imgGT = cv.imread(r"D:\training_set_v2\Medieval720_Test2\diffuse\gt\GTDiffuse.0005.exr",cv.IMREAD_UNCHANGED)
imgWarp = cv.imread(r"D:\training_set_v2\Medieval720_Test2\diffuse\no_hole_noaa\DemoSceneWrapDiffuse.0005.exr",cv.IMREAD_UNCHANGED)


result = imgGT - imgWarp

result = np.abs(result)

thres = np.mean(result)+3*np.std(result)



mask = result.copy()



mask[np.abs(result) > thres] = 10
mask[np.abs(result) <= thres] = 0

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        for k in range(3):
            if mask[i][j][k] == 10:
                mask[i][j] = np.array([10,10,10])
                break
            # if mask[i][j][k] != 10:
            #     mask[i][j] = np.array([0,0,0])
            #     break

cv.imwrite("test_good_mask.exr", mask)