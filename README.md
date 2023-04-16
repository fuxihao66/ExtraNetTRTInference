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

PS: If you want to test performance, you need to add **cudaDeviceSynchronize()** before executing inference.

Example:
```
cudaDeviceSynchronize();
std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

for (int i = 0; i < 1000; i++){
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }
}

std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
```
