# OnnxGeneration

This project is used to generate onnx for tensorrt infer.

**Currently, only CUDA11.0 is supported. And CUDNN is also required to run TRT inference.**

###  Onnx Generate

run deploy.py

Optional:

* prepareDataForTRT.py:   transfer exr image to txt for trt inference

* textToImage.py: trt result to exr image



### TRT Inference

open **sampleOnnxMNIST** project in [TensorRT-8.2.5.1.zip](https://drive.google.com/file/d/1-YDmLoIVlFW_K6msqLZF0RKZ8N3jTO5C/view?usp=sharing)

1. move the generated onnx file to TensorRT-8.2.5.1\data\mnist
2. set params.onnxFileName
3. add TensorRT-8.2.5.1\lib to Environment Variables **PATH**
