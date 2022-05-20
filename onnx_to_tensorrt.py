#!/usr/bin/env python2
#
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import print_function

import numpy as np
import tensorrt as trt
from utils import ToneSimple,DeToneSimple,ReadData
import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv
from config import mdevice


import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder:
            builder.fp16_mode = True
            with builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
                builder.max_workspace_size = 1 << 30 # 256MiB
                #builder.max_batch_size = 1
                # Parse model file
                if not os.path.exists(onnx_file_path):
                    print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(onnx_file_path))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    if not parser.parse(model.read()):
                        print ('ERROR: Failed to parse the ONNX file.')
                        for error in range(parser.num_errors):
                            print (parser.get_error(error))
                        return None
                # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
                print(network.get_input(0).shape)
                print(network.get_input(1).shape)
                print(network.get_input(2).shape)
                print(network.get_input(3).shape)
                print('Completed parsing of ONNX file')
                print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
                engine = builder.build_cuda_engine(network)
                print("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
def load_engine(engine_file_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
def Tensor2NP(t):
    res=t.squeeze(0).cpu().numpy().transpose([1,2,0])
    res=cv.cvtColor(res,cv.COLOR_RGB2BGR)
    res = DeToneSimple(res)
    return res
def NP2NP(t):
    res=t[0].transpose([1,2,0])
    res=cv.cvtColor(res,cv.COLOR_RGB2BGR)
    res = DeToneSimple(res)
    return res
from Small_UnetGated import UNetLWGated_FULL
import torch
# model = UNetLWGated_FULL(18, 3)
# model.load_state_dict(torch.load("totalModel.140.pth.tar")["state_dict"])
# model = model.to(mdevice)
# model.eval()

with get_engine("./UNetGated_UEIntegrate.onnx","./UNetLWGated__trt_fp16.pb") as engine, engine.create_execution_context() as context:
    with torch.no_grad():
        img, img_2, img_3, occ_warp_img, woCheckimg, woCheckimg_2, woCheckimg_3, labelimg, metalic, Roughnessimg, Depthimg, Normalimg = ReadData(
            "G:/TrainingSetCompressed/bunker_compressed/BK_Train_1/compressed.0168.npz")

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

        # hisBuffer = torch.tensor(hisBuffer).to(mdevice)
        input = torch.tensor(input).unsqueeze(0).to(mdevice)
        # features = torch.tensor(features).unsqueeze(0).to(mdevice)

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        inputs[0].host = input.cpu().numpy().copy()
        # inputs[0].host = input.cpu().numpy().copy()
        # inputs[1].host = features.cpu().numpy().copy()
        # inputs[2].host = finalMask.cpu().numpy().copy()

        '''
        inputs[0].host=np.random.randn(1, 6, 720, 1280).astype(np.float32)
        inputs[1].host=np.random.randn(1, 6, 720, 1280).astype(np.float32)
        inputs[2].host=np.random.randn(1, 6, 720, 1280).astype(np.float32)
        inputs[3].host=np.random.randn(3, 4, 720, 1280).astype(np.float32)
        '''
        [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        output=output.reshape(1,3,720,1280)

        # gt = model(input,features,finalMask,hisBuffer)
        # print(torch.mean(torch.abs(gt-torch.from_numpy(output).to(mdevice))))

        cv.imwrite("testResult.exr", NP2NP(output))
        # cv.imwrite("./resTrain/res train %s.exr" % ("gt"), Tensor2NP(gt))