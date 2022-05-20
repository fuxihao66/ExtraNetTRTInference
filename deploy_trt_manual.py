from Small_UnetGated import UNetLWGated,ConvHis
import torch
import tensorrt as trt
import common
import numpy as np
import torch.nn as nn
from config import mdevice
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
model_his = ConvHis()
model_his.load_state_dict(torch.load("totalModel.140.pth.tar")["state_dict"],strict=False)
model_his=model_his.to(mdevice).eval()
for key in model_his.state_dict():
    print(key)
print(model_his.state_dict()["convHis3.7.running_mean"])
def build_engine():
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = common.GiB(1)
        input_tensor = network.add_input(name="test",dtype=trt.float32,shape=(1,1,2,2))
        upsample1=network.add_resize(input=input_tensor)
        upsample1.resize_mode = trt.ResizeMode.NEAREST
        upsample1.shape=(1,1,4,4)
        upsample1.scales=[1,1,2,2]
        network.mark_output(tensor=upsample1.get_output(0))
        return builder.build_cuda_engine(network)
with build_engine() as engine:
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    inputs[0].host=np.arange(1,5).reshape((1,1,2,2)).astype(np.float32)
    m_upsample=nn.Upsample(scale_factor=2, mode='nearest')
    print(inputs[0].host)
    print(m_upsample(torch.from_numpy(inputs[0].host)))
    with engine.create_execution_context() as context:
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print(output.reshape(1,1,4,4))
