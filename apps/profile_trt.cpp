#include <vector>
#include<fstream>
#include <iostream>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cmath>

#include "trt_engine.hpp"

#define input_PATH "results/cuda_preprocess/home_input_V1.bin"
#define model_PATH "models/yolov8n_cpp_fp16_profile.engine"


int main()
{

    float* host_input=nullptr;
    float* device_input=nullptr;

    std::ifstream file(input_PATH,std::ios::binary|std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "文件打开失败！" << std::endl;
        return false;
    }

    int input_size=file.tellg();
    file.seekg(0,std::ios::beg);

    cudaError_t err=cudaHostAlloc(
        reinterpret_cast<void**>(&host_input),
        input_size,
        cudaHostAllocMapped
    );
    if(err!=cudaSuccess)
    {
        std::cerr<<"cudaHostAlloc fail:"<<cudaGetErrorString(err)<<std::endl;

        return -1;
    }

    file.read(
        reinterpret_cast<char*>(host_input),
        input_size
    );

    cudaError_t err1=cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&device_input),
        host_input,
        0
    );
    if(err1!=cudaSuccess)
    {
        std::cerr<<"cudaHostGetDevicePointer fail:"<<cudaGetErrorString(err1)<<std::endl;

        return -1;
    }


    trtengine trtengine;
    trtengine.batch=1;
    nvinfer1::Dims4 dim{trtengine.batch,3,640,640};
    if(!trtengine.load(model_PATH))
        {
            std::cout<<"模型加载"<<std::endl;
            return -1;
        }

    if(!trtengine.createContext())
        {
            std::cout<<"创建context"<<std::endl;
            return -1;
        }

    if(!trtengine.getIOname())
        {
            return -1;
        }

    if(!trtengine.setInputShape(dim))
        {
            return -1;
        }

    if(!trtengine.setupIO())
        {
            return -1;
        }

    cudaMalloc(
        &trtengine.device_output,
        trtengine.output_bytes_
    );

    RuntimeTiming timing;

    for(int i=0;i<10;i++)
    {
        
        if(!trtengine.inferDevice(
                device_input,
                &timing))
            {
                std::cout<<"inferDevice"<<std::endl;
                return -1;
            }
    }

}