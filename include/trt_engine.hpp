#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include<iostream>
#include <NvInfer.h>
#include<vector>

#define cuda_check(call)                 \
do {                                     \
    cudaError_t error=(call);            \
    if(error !=cudaSuccess)              \
    {                                    \
        std::cerr                        \
            <<"[CUDA ERROR]"             \
            <<cudaGetErrorString(error)  \
            <<"\n file:"<<__FILE__       \
            <<"\n line:"<<__LINE__       \
            <<std::endl;                 \
        std::exit(EXIT_FAILURE);         \
    }                                    \
} while (0)                              \



struct RuntimeTiming{

    float h2d_ms=0.0f;
    float trt_ms=0.0f;
    float d2h_ms=0.0f;

};

class trtengine{

public:
    bool load(const std::string& engine_path);
    bool getIOname();
    bool createContext();
    bool setInputShape(const nvinfer1::Dims4& dims);
    bool setupIO();

    bool infer(
        const float*  host_input,
        float* host_output,
        RuntimeTiming* timeing
    );
    bool inferDevice(
        float* device_cudaprepro_input,
        RuntimeTiming* timeing
    );
    bool inferfram(
        float* device_cudaprepro_input,
        float* device_fram_output
    );

    bool prepare(const nvinfer1::Dims4& dims);

    bool readFloatBinary(
       const std::string& path,
       std::vector<float>& input
    );

    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* cudaengine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};

    cudaStream_t stream=nullptr;

    void* device_input=nullptr;
    void* device_output=nullptr;
        
    std::string input_name_;
    std::string output_name_;

    size_t input_elements_;
    size_t output_elements_;

    size_t input_bytes_;
    size_t output_bytes_;   
    
    cudaEvent_t e0_{nullptr};
    cudaEvent_t e1_{nullptr};
    cudaEvent_t e2_{nullptr};
    cudaEvent_t e3_{nullptr};

    int batch;
};
