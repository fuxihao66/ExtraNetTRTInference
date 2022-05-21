# OnnxGeneration

use to generate onnx for tensorrt infer



###  Onnx Generate

run deploy.py

Optional:

* prepareDataForTRT.py:   transfer exr image to txt for trt inference

* textToImage.py: trt result to exr image



### TRT Inference

open **sampleOnnxMNIST** project in [TensorRT-7.2.2.3.zip](https://drive.google.com/file/d/15Cc4JV-o0zw4f_8qT85pHBQx51NFKmT7/view?usp=sharing)

1. move the generated onnx file to TensorRT-7.2.2.3\data\mnist
2. set params.onnxFileName
