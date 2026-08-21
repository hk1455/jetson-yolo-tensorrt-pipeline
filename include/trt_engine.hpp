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

class trtengine{

public:
    bool load(const std::string& engine_path);
    void printTensorInfo();
    bool createContext();
    bool setInputShape(const nvinfer1::Dims4& dims);
    void printRuntime();
    bool infer(int batch,const std::string& out_BIN_PATH);

    bool readFloatBinary(
        const std::string& path,
        size_t expected_elements
    );

    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* cudaengine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
        
    std::vector<std::string> input_name;
    std::vector<std::string> output_name;

    //element count
    std::vector<float> host_input;
    std::vector<float> host_output;

    int input_datasize;
    int output_datasize;
    int batch;
};

