import torch
import cv2 as cv
from config import mdevice
import numpy as np
from utils import ToneSimple,DeToneSimple,ReadData
import math
result = np.zeros([3, 720, 1280],dtype=np.float32)
with open("result.txt", "r") as inputFile:
    buffers = inputFile.read().split(" ")
    index = 0
    for i in range(3):
        for j in range(720):
            for k in range(1280):
                result[i,j,k] = math.exp(float(buffers[index]))-1.
                index += 1
result = result.transpose([1, 2, 0])
cv.imwrite("result.exr", result)