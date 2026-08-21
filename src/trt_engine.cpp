
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
//打印cudaengine的输入输出张量信息，同时记录输入输出的名字
void trtengine::printTensorInfo(){
    int tensor_size=cudaengine_->getNbIOTensors();


    for(int i=0;i<tensor_size;i++){
        const char *name=cudaengine_->getIOTensorName(i);

        nvinfer1::TensorIOMode mode=cudaengine_->getTensorIOMode(name);

        if(mode==nvinfer1::TensorIOMode::kINPUT)
            input_name.push_back(name);
        if(mode==nvinfer1::TensorIOMode::kOUTPUT)
            output_name.push_back(name);

        nvinfer1::DataType datatype=cudaengine_->getTensorDataType(name);
        nvinfer1::Dims dims=cudaengine_->getTensorShape(name);
        std::cout<<name<<std::endl;
        std::cout<<ioModeToString(mode)<<std::endl;
        std::cout<<dataTypetoString(datatype)<<std::endl;
        std::cout<<dimStoString(dims)<<std::endl;
    };



};

//根据cudaengine创建本次执行使用的context
bool trtengine::createContext(){
    context_=cudaengine_->createExecutionContext();
    if(!context_)
    {
        std::cout<<"fail to create context";

        return 0;
    }
    return 1;

};

//给context设置inputshape
bool trtengine::setInputShape(const nvinfer1::Dims4& dims){
 for(size_t i=0;i<input_name.size();i++)
   { 
        bool suc=context_->setInputShape
        (
            input_name[i].c_str(),
            dims
        );

        if(!suc)
        {
            std::cout<<"fail to set context";

            return 0;
        }
    }
    return 1;
};

//获取context运行时的shape，并计算input/output tensor的元素数量和buffer的字节数量
void trtengine::printRuntime(){
    for(size_t i=0;i<input_name.size();i++)
   { 
        nvinfer1::Dims context_dims=context_->getTensorShape(input_name[i].c_str());
        nvinfer1::DataType datatype=cudaengine_->getTensorDataType(input_name[i].c_str());

        int element_count=volume(context_dims);

        host_input.resize(volume(context_dims));
        input_datasize=datasize(datatype)*element_count;

        std::cout<<dimStoString(context_dims)<<std::endl;
        std::cout<<"Element : "<<element_count<<std::endl;

        std::cout<<dataTypetoString(datatype)<<": ";
        std::cout<<input_datasize<<std::endl;
    }

    for(size_t i=0;i<output_name.size();i++)
    { 
        nvinfer1::Dims context_dims=context_->getTensorShape(output_name[i].c_str());
        nvinfer1::DataType datatype=cudaengine_->getTensorDataType(output_name[i].c_str());
        
        int element_count=volume(context_dims);

        host_output.resize(element_count);
        output_datasize=datasize(datatype)*element_count;
        
        std::cout<<dimStoString(context_dims)<<std::endl;
        std::cout<<"Element : "<<element_count<<std::endl;

        std::cout<<dataTypetoString(datatype)<<": ";
        std::cout<< output_datasize<<std::endl;

    }
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

bool trtengine::readFloatBinary(const std::string& path,size_t expected_elements)
{
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "文件打开失败！" << std::endl;
        return false;
    }

    expected_elements=file.tellg();
    file.seekg(0,std::ios::beg);

    host_input.resize(expected_elements/sizeof(float));
    file.read (reinterpret_cast<char*>(host_input.data()),expected_elements);

    return true;
}

bool trtengine::infer(int batch,const std::string& out_BIN_PATH){

    setInputShape(nvinfer1::Dims4 {batch,3,640,640});
    printRuntime();
    std::fill(
        host_output.begin(),
        host_output.end(),
        -999.0f
    );

    void* device_input=nullptr;
    void* device_output=nullptr;

    cuda_check(
            cudaMalloc(
            &device_input,
            input_datasize
        )
    );

    cuda_check(
            cudaMalloc(
            &device_output,
            output_datasize
        )
    );   

    cudaStream_t stream=nullptr;

    cuda_check(
        cudaStreamCreate(&stream)
    );

    cuda_check(
        cudaMemcpyAsync(
            device_input,
            host_input.data(),
            input_datasize,
            cudaMemcpyHostToDevice,
            stream
        )
    );

    if(
        !context_->setTensorAddress(
        input_name[0].c_str(),
        device_input
    ))
    {
        std::cerr<<"fail to settensorAddress input";
    }
    if(!context_->setTensorAddress(
        output_name[0].c_str(),
        device_output
    ))
    {
        std::cerr<<"fail to settensorAddress output";
    }

    bool success=context_->enqueueV3(stream);

    if(!success){
        std::cerr<<"TensorRT enqueueV3 failed";
    }

    cuda_check(
        cudaMemcpyAsync(
            host_output.data(),
            device_output,
            output_datasize,
            cudaMemcpyDeviceToHost,
            stream
        )
    );

    cudaStreamSynchronize(stream);

    // size_t single_output_elements = 84 * 8400;
    // std::cout << trtengine.host_output[0] << '\n';
    // std::cout << trtengine.host_output[single_output_elements] << '\n';
    // std::cout << trtengine.host_output[2 * single_output_elements] << '\n';
    // std::cout << trtengine.host_output[3 * single_output_elements] << '\n';

    std::cout << "output[0]      = " << host_output.at(0) << '\n';
    std::cout << "output[1]      = " << host_output.at(1) << '\n';
    std::cout << "output[10]     = " << host_output.at(10) << '\n';
    std::cout << "output[100]    = " << host_output.at(100) << '\n';
    std::cout << "output[1000]   = " << host_output.at(1000) << '\n';
    std::cout << "output[mid]    = " << host_output.at(705600 / 2) << '\n';
    std::cout << "output[last]   = " << host_output.at(705600 - 1) << '\n';

    // printTensorStats(trtengine.host_output);

    std::ofstream ofs(out_BIN_PATH,std::ios::binary);

    ofs.write(
        reinterpret_cast<const char*>(host_output.data()), 
        host_output.size() * sizeof(float)
    );
};
