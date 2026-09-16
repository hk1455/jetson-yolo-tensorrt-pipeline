
#include <fstream>
#include<vector>
#include<string>
#include <NvInfer.h>
#include<iostream>
#include <algorithm>
#include<cuda_runtime.h>
#include<cstdlib>
#include<cmath>
#include "../include/trt_engine.hpp"
#include <NvInfer.h>
#include<iostream>
#include<fstream>
#include <algorithm>
#include<cuda_runtime.h>
#include<cstdlib>
#include<cmath>

class Logger:public nvinfer1::ILogger{
    public:
        void log(
            Severity severity,
            const char* msg
        )noexcept override
    {
        if(severity<=Severity::kWARNING)
        std::cout<<"{TensorRT}"<<msg<<std::endl;
    }

};

std::string ioModeToString(nvinfer1::TensorIOMode mode)
{
//! Tensor is not an input or output.
    if(mode==nvinfer1::TensorIOMode::kNONE)
        return "NONE";
//! Tensor is input to the engine.
    else if(mode==nvinfer1::TensorIOMode::kINPUT)
        return "INPUT";
//! Tensor is output by the engine.
    else if(mode==nvinfer1::TensorIOMode::kOUTPUT)
        return "OUTPUT";

    return "UNKNOWN";
};


std::string dataTypetoString(nvinfer1::DataType datatype)
{
    
        //! 32-bit floating point format.
    if(datatype==nvinfer1::DataType::kFLOAT) 
        return "32-bit";
    //! IEEE 16-bit floating-point format -- has a 5 bit exponent and 11 bit significand.
    else if(datatype==nvinfer1::DataType:: kHALF) 
        return "IEEE 16-bit";
    //! Signed 8-bit integer representing a quantized floating-point value.
    else if(datatype==nvinfer1::DataType:: kINT8) 
        return "Signed 8-bit";
    //! Signed 32-bit integer format.
    else if(datatype==nvinfer1::DataType::kINT32) 
        return "Signed 32-bit";
    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
    else if(datatype==nvinfer1::DataType::kBOOL) 
        return "8-bit boolean";
    
    return "unknown";
};

std::string dimStoString(const nvinfer1::Dims& dims){
    std::string dimstring;
    dimstring+="[";
    for(int i=0;i<dims.nbDims;i++)
    {
        if(i<dims.nbDims-1)
            {
                dimstring+=std::to_string(dims.d[i]);
                dimstring+=",";
            }
        else
            {
                dimstring+=std::to_string(dims.d[i]);
                dimstring+="]";
            }
    }

    return dimstring;
};

int volume(const nvinfer1::Dims& dims){
    int element=1;
//dims.nbDims是表示这个张量有多少维度，dims.d[i]表示第i个维度的大小
    for(int i=0;i<dims.nbDims;i++)
    {
      
        element=element*dims.d[i];
            
    }
    return element;
};

int datasize(nvinfer1::DataType datatype)
{
    if(datatype==nvinfer1::DataType::kFLOAT) 
        return 4;
    //! IEEE 16-bit floating-point format -- has a 5 bit exponent and 11 bit significand.
    else if(datatype==nvinfer1::DataType:: kHALF) 
        return 2;
    //! Signed 8-bit integer representing a quantized floating-point value.
    else if(datatype==nvinfer1::DataType:: kINT8) 
        return 1;
    //! Signed 32-bit integer format.
    else if(datatype==nvinfer1::DataType::kINT32) 
        return 4;
    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
    else if(datatype==nvinfer1::DataType::kBOOL) 
        return 1;

};


static Logger logger;
//创建runtime，读取.engine文件序列化数据。
//然后再用deserializeCudaEngine反序列化得到cudaengine
bool trtengine::load(const std::string& engine_path)

{

    runtime_=nvinfer1::createInferRuntime(logger);

    std::ifstream file(
        engine_path,std::ios::binary
    );

    if(!file){
        std::cerr<<"fail to open file"<<std::endl;
        return 0;
    }


    file.seekg(0,std::ios::end);
    std::streamsize file_size=file.tellg();
    std::vector<char> engine_data(file_size);

    file.seekg(0,std::ios::beg);

    file.read(
        engine_data.data(),
        file_size
    );


    cudaengine_=runtime_->deserializeCudaEngine(
            engine_data.data(),
            engine_data.size()
    );

    if(!cudaengine_){
        std::cerr<<"fail to run engine"<<std::endl;
        return 0;
    }
    return 1;
};
//记录输入输出的名字
bool trtengine::getIOname(){
    int tensor_size=cudaengine_->getNbIOTensors();


    for(int i=0;i<tensor_size;i++){
        const char *name=cudaengine_->getIOTensorName(i);

        nvinfer1::TensorIOMode mode=cudaengine_->getTensorIOMode(name);

        if(mode==nvinfer1::TensorIOMode::kINPUT)
            input_name_=name;
        if(mode==nvinfer1::TensorIOMode::kOUTPUT)
            output_name_=name;
    };

    return true;
};

//根据cudaengine创建本次执行使用的context
bool trtengine::createContext(){
    context_=cudaengine_->createExecutionContext();
    if(!context_)
    {
        std::cout<<"fail to create context";

        return false;
    }
    return true;
};

//给context设置inputshape
bool trtengine::setInputShape(const nvinfer1::Dims4& dims){

        bool suc=context_->setInputShape
        (
            input_name_.c_str(),
            dims
        );

        if(!suc)
        {
            std::cout<<"fail to set context";

            return false;
        }
    return true;
};

//获取获取input/output的数量和姓名，context运行时的shape，并计算input/output tensor的元素数量和buffer的字节数量
bool trtengine::setupIO(){
        //根据inputname获取输入的shape
        nvinfer1::Dims context_dims=context_->getTensorShape(input_name_.c_str());
        //根据inputname得到数据类型
        nvinfer1::DataType datatype=cudaengine_->getTensorDataType(input_name_.c_str());
        //根据shape得到element_count
        int element_count=volume(context_dims);
        //根据element_count和datatype得到输入的大小
        input_bytes_=datasize(datatype)*element_count;
        input_elements_=input_bytes_/datasize(datatype);

        //output流程同上
        nvinfer1::Dims out_context_dims=context_->getTensorShape(output_name_.c_str());
        nvinfer1::DataType out_datatype=cudaengine_->getTensorDataType(output_name_.c_str());
        
        int out_element_count=volume(out_context_dims);
        output_bytes_=datasize(out_datatype)*out_element_count;
        output_elements_=output_bytes_/datasize(datatype);
    
        return true;
};




void printTensorStats(const std::vector<float>& data)
{
    float min_v = data[0];
    float max_v = data[0];
    double sum = 0.0;
    int nan_count = 0;
    int inf_count = 0;
    for (float x : data) {
        min_v = std::min(min_v, x);
        max_v = std::max(max_v, x);
        sum += x;
        if (std::isnan(x)) ++nan_count;
        if (std::isinf(x)) ++inf_count;
    }
    std::cout << "min: " << min_v << '\n';
    std::cout << "max: " << max_v << '\n';
    std::cout << "mean: " << sum / data.size() << '\n';
    std::cout << "NaN: " << nan_count << '\n';
    std::cout << "Inf: " << inf_count << '\n';
}

bool trtengine::readFloatBinary(const std::string& path,std::vector<float>& input)
{
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "文件打开失败！" << std::endl;
        return false;
    }

    int bytes=file.tellg();
    file.seekg(0,std::ios::beg);

    input.resize(bytes/sizeof(float));
    file.read (reinterpret_cast<char*>(input.data()),bytes);

    return true;
}

bool trtengine::prepare(const nvinfer1::Dims4& dims)
{
    if(!getIOname()){
        return false;
    }
    if(!setInputShape(dims)){
        return false;
    }
    if(!setupIO()){
        return false;
    }

    cuda_check(
            cudaMalloc(
            &device_input,
            input_bytes_
        )
    );

    cuda_check(
            cudaMalloc(
            &device_output,
            output_bytes_
        )
    );   

    cuda_check(
        cudaStreamCreate(&stream)
    );
     if(
        !context_->setTensorAddress(
        input_name_.c_str(),
        device_input
    ))
    {
        std::cerr<<"fail to settensorAddress input";
        return false;
    }
    if(!context_->setTensorAddress(
        output_name_.c_str(),
        device_output
    ))
    {
        std::cerr<<"fail to settensorAddress output";
        return false;
    }

    cudaEventCreate(&e0_);
    cudaEventCreate(&e1_);
    cudaEventCreate(&e2_);
    cudaEventCreate(&e3_);

    return true;
};


bool trtengine::infer(
        const float*  host_input,
        float*  host_output,
        RuntimeTiming* timeing
    )
{

    cudaEventRecord(
        e0_,
        stream
    );

    cuda_check(
        cudaMemcpyAsync(
            device_input,
            host_input,
            input_bytes_,
            cudaMemcpyHostToDevice,
            stream
        )
    );

    cudaEventRecord(
        e1_,
        stream
    );

    if(!context_->enqueueV3(stream)){
        std::cerr<<"TensorRT enqueueV3 failed";
        return false;
    }

    cudaEventRecord(
        e2_,
        stream
    );

    cuda_check(
        cudaMemcpyAsync(
            host_output,
            device_output,
            output_bytes_,
            cudaMemcpyDeviceToHost,
            stream
        )
    );
    cudaEventRecord(
        e3_,
        stream
    );

    cudaEventSynchronize(e3_);

    if(timeing!=nullptr)
    {
        cudaEventElapsedTime(
            &timeing->h2d_ms,
            e0_,
            e1_
        );
        cudaEventElapsedTime(
            &timeing->trt_ms,
            e1_,
            e2_
        );
        cudaEventElapsedTime(
            &timeing->d2h_ms,
            e2_,
            e3_
        );
    }



    
    return true;

    // size_t single_output_elements = 84 * 8400;
    // std::cout << trtengine.host_output[0] << '\n';
    // std::cout << trtengine.host_output[single_output_elements] << '\n';
    // std::cout << trtengine.host_output[2 * single_output_elements] << '\n';
    // std::cout << trtengine.host_output[3 * single_output_elements] << '\n';
    // printTensorStats(trtengine.host_output);

    // std::ofstream ofs(out_BIN_PATH,std::ios::binary);

    // ofs.write(
    //     reinterpret_cast<const char*>(host_output.data()), 
    //     host_output.size() * sizeof(float)
    // );
};


#include <chrono>
using Clock=std::chrono::steady_clock;

bool trtengine::inferDevice(
        float*  device_cudaprepro_input,
        RuntimeTiming* timeing
        
){
    
     if(
        !context_->setTensorAddress(
        input_name_.c_str(),
        device_cudaprepro_input
    ))
    {
        std::cerr<<"fail to settensorAddress input";
        return false;
    }
    if(!context_->setTensorAddress(
        output_name_.c_str(),
        device_output
    ))
    {
        std::cerr<<"fail to settensorAddress output";
        return false;
    }
    

    if(!context_->enqueueV3(stream)){
        std::cerr<<"TensorRT enqueueV3 failed";
        return false;
    }
    
    return true;

}

bool trtengine::inferfram(
        float* device_cudaprepro_input,
        float* device_fram_output
){
         if(
        !context_->setTensorAddress(
        input_name_.c_str(),
        device_cudaprepro_input
    ))
    {
        std::cerr<<"fail to settensorAddress input";
        return false;
    }
    if(!context_->setTensorAddress(
        output_name_.c_str(),
        device_fram_output
    ))
    {
        std::cerr<<"fail to settensorAddress output";
        return false;
    }
    

    if(!context_->enqueueV3(stream)){
        std::cerr<<"TensorRT enqueueV3 failed";
        return false;
    }
    
    return true;
}