import Small_UnetGated
import cv2
import numpy as np
from utils import  ToneSimple, DeToneSimple,ImgReadWithPrefix, ReadData
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.utils import save_image
from torch import optim
import Losses
import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
ScenePrefix = "DemoScene"
WarpPrefix = "Wrap"
GtPrefix = "GT"
NormalPrefix = "WorldNormal"
DepthPrefix = "SceneDepth"
MetalicPrefix = "Metallic"
RoughPrefix = "Roughness"

def MergeRange(idx, inPath, outPath):
    newIdx = str(idx).zfill(4)
    img = cv2.imread(inPath + "/wrap_res/" + ScenePrefix + WarpPrefix + ".{}.1.exr".format(newIdx),
                     cv2.IMREAD_UNCHANGED)
    img3 = cv2.imread(inPath + "/wrap_res/" + ScenePrefix + WarpPrefix + ".{}.3.exr".format(newIdx),
                      cv2.IMREAD_UNCHANGED)
    img5 = cv2.imread(inPath + "/wrap_res/" + ScenePrefix + WarpPrefix + ".{}.5.exr".format(newIdx),
                      cv2.IMREAD_UNCHANGED)
    imgOcc = cv2.imread(inPath + "/occ/" + ScenePrefix + WarpPrefix + ".{}.1.exr".format(newIdx),
                        cv2.IMREAD_UNCHANGED)
    img_no_hole = cv2.imread(inPath + "/wrap_no_hole/" + ScenePrefix + WarpPrefix + ".{}.1.exr".format(newIdx),
                             cv2.IMREAD_UNCHANGED)
    img_no_hole3 = cv2.imread(inPath + "/wrap_no_hole/" + ScenePrefix + WarpPrefix + ".{}.3.exr".format(newIdx),
                              cv2.IMREAD_UNCHANGED)
    img_no_hole5 = cv2.imread(inPath + "/wrap_no_hole/" + ScenePrefix + WarpPrefix + ".{}.5.exr".format(newIdx),
                              cv2.IMREAD_UNCHANGED)
    if "taa" in inPath:
        gt = cv2.imread(inPath + "/GT/" + GtPrefix + ".{}.exr".format(newIdx), cv2.IMREAD_UNCHANGED)
    else:
        gt = cv2.imread(inPath + "/GT/" + ScenePrefix + GtPrefix + ".{}.exr".format(newIdx), cv2.IMREAD_UNCHANGED)
    metalic = cv2.imread(inPath + "/" + ScenePrefix + MetalicPrefix + ".{}.exr".format(newIdx),
                         cv2.IMREAD_UNCHANGED)[:, :, 0:1]
    roughness = cv2.imread(inPath + "/" + ScenePrefix + RoughPrefix + ".{}.exr".format(newIdx),
                           cv2.IMREAD_UNCHANGED)[:, :, 0:1]
    depth = cv2.imread(inPath + "/" + ScenePrefix + DepthPrefix + ".{}.exr".format(newIdx), cv2.IMREAD_UNCHANGED)[:,
            :, 0:1]
    normal = cv2.imread(inPath + "/" + ScenePrefix + NormalPrefix + ".{}.exr".format(newIdx), cv2.IMREAD_UNCHANGED)

    res = np.concatenate(
        [img, img3, img5, imgOcc, img_no_hole, img_no_hole3, img_no_hole5, gt, metalic, roughness, depth, normal],
        axis=2)
    res = res.astype(np.float16)
    np.save(outPath + 'compressed.{}'.format(newIdx), res)

