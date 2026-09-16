// #include <NvInfer.h>
// #include "../include/trt_engine.hpp"
// #include<iostream>
// #include<fstream>
// #include <algorithm>
// #include<cuda_runtime.h>
// #include<cstdlib>
// #include<cmath>

// const char* MODEL_PATH="models/yolov8n_cpp_fp32.engine";

// const char* In_BIN_PATH="results/ort/input_b3.bin";
// const char* out_BIN_PATH="results/ort/output_b3.bin";

// #define cuda_check(call)                 \
// do {                                     \
//     cudaError_t error=(call);            \
//     if(error !=cudaSuccess)              \
//     {                                    \
//         std::cerr                        \
//             <<"[CUDA ERROR]"             \
//             <<cudaGetErrorString(error)  \ 
//             <<"\n file:"<<__FILE__       \ 
//             <<"\n line:"<<__LINE__       \
//             <<std::endl;                 \
//         std::exit(EXIT_FAILURE);         \
//     }                                    \ 
// } while (0)                              \

// void printTensorStats(const std::vector<float>& data)
// {
//     float min_v = data[0];
//     float max_v = data[0];
//     double sum = 0.0;
//     int nan_count = 0;
//     int inf_count = 0;
//     for (float x : data) {
//         min_v = std::min(min_v, x);
//         max_v = std::max(max_v, x);
//         sum += x;
//         if (std::isnan(x)) ++nan_count;
//         if (std::isinf(x)) ++inf_count;
//     }
//     std::cout << "min: " << min_v << '\n';
//     std::cout << "max: " << max_v << '\n';
//     std::cout << "mean: " << sum / data.size() << '\n';
//     std::cout << "NaN: " << nan_count << '\n';
//     std::cout << "Inf: " << inf_count << '\n';
// }

// bool trtengine::readFloatBinary(const std::string& path,size_t expected_elements)
// {
//     std::ifstream file(path,std::ios::binary|std::ios::ate);
//     if (!file.is_open()) {
//         std::cerr << "文件打开失败！" << std::endl;
//         return -1;
//     }

//     expected_elements=file.tellg();
//     file.seekg(0,std::ios::beg);

//     host_input.resize(expected_elements/sizeof(float));
//     file.read (reinterpret_cast<char*>(host_input.data()),expected_elements);

// }

// bool trtengine::infer(int batch,const std::string& out_BIN_PATH){

//     setInputShape(nvinfer1::Dims4 {batch,3,640,640});
//     printRuntime();
//     std::fill(
//         host_output.begin(),
//         host_output.end(),
//         -999.0f
//     );

//     void* device_input=nullptr;
//     void* device_output=nullptr;

//     cuda_check(
//             cudaMalloc(
//             &device_input,
//             input_datasize
//         )
//     );

//     cuda_check(
//             cudaMalloc(
//             &device_output,
//             output_datasize
//         )
//     );   

//     cudaStream_t stream=nullptr;

//     cuda_check(
//         cudaStreamCreate(&stream)
//     );

//     cuda_check(
//         cudaMemcpyAsync(
//             device_input,
//             host_input.data(),
//             input_datasize,
//             cudaMemcpyHostToDevice,
//             stream
//         )
//     );

//     if(
//         !context_->setTensorAddress(
//         input_name[0].c_str(),
//         device_input
//     ))
//     {
//         std::cerr<<"fail to settensorAddress input";
//     }
//     if(!context_->setTensorAddress(
//         output_name[0].c_str(),
//         device_output
//     ))
//     {
//         std::cerr<<"fail to settensorAddress output";
//     }

//     bool success=context_->enqueueV3(stream);

//     if(!success){
//         std::cerr<<"TensorRT enqueueV3 failed";
//     }

//     cuda_check(
//         cudaMemcpyAsync(
//             host_output.data(),
//             device_output,
//             output_datasize,
//             cudaMemcpyDeviceToHost,
//             stream
//         )
//     );

//     cudaStreamSynchronize(stream);

//     // size_t single_output_elements = 84 * 8400;
//     // std::cout << trtengine.host_output[0] << '\n';
//     // std::cout << trtengine.host_output[single_output_elements] << '\n';
//     // std::cout << trtengine.host_output[2 * single_output_elements] << '\n';
//     // std::cout << trtengine.host_output[3 * single_output_elements] << '\n';

//     std::cout << "output[0]      = " << host_output.at(0) << '\n';
//     std::cout << "output[1]      = " << host_output.at(1) << '\n';
//     std::cout << "output[10]     = " << host_output.at(10) << '\n';
//     std::cout << "output[100]    = " << host_output.at(100) << '\n';
//     std::cout << "output[1000]   = " << host_output.at(1000) << '\n';
//     std::cout << "output[mid]    = " << host_output.at(705600 / 2) << '\n';
//     std::cout << "output[last]   = " << host_output.at(705600 - 1) << '\n';

//     // printTensorStats(trtengine.host_output);

//     std::ofstream ofs(out_BIN_PATH,std::ios::binary);

//     ofs.write(
//         reinterpret_cast<const char*>(host_output.data()), 
//         host_output.size() * sizeof(float)
//     );
// };


// // int main()
// // {
// //     trtengine trtengine;
// //     trtengine.load(MODEL_PATH);
// //     trtengine.printTensorInfo();

// //     trtengine.createContext();
// //     trtengine.setInputShape(nvinfer1::Dims4 {3,3,640,640});
// //     trtengine.printRuntime();


// //     std::vector<float> host_input;
// //     size_t in_size;
// //     //size_t out_size;

// //     //std::vector<float> host_output;

// //     readFloatBinary(In_BIN_PATH,host_input,in_size);
// //     //readFloatBinary(out_BIN_PATH,host_output,out_size);

// //     trtengine.host_input=host_input;
// //     //trtengine.host_output=host_output;

// //     // std::fill(
// //     //     trtengine.host_input.begin(),
// //     //     trtengine.host_input.end(),
// //     //     0.5f
// //     // );

// //     std::fill(
// //         trtengine.host_output.begin(),
// //         trtengine.host_output.end(),
// //         -999.0f
// //     );

// //     void* device_input=nullptr;
// //     void* device_output=nullptr;

// //     cuda_check(
// //             cudaMalloc(
// //             &device_input,
// //             trtengine.input_datasize
// //         )
// //     );

// //     cuda_check(
// //             cudaMalloc(
// //             &device_output,
// //             trtengine.output_datasize
// //         )
// //     );   

// //     cudaStream_t stream=nullptr;

// //     cuda_check(
// //         cudaStreamCreate(&stream)
// //     );

// //     cuda_check(
// //         cudaMemcpyAsync(
// //             device_input,
// //             trtengine.host_input.data(),
// //             trtengine.input_datasize,
// //             cudaMemcpyHostToDevice,
// //             stream
// //         )
// //     );

// //     if(
// //         !trtengine.context_->setTensorAddress(
// //         trtengine.input_name[0].c_str(),
// //         device_input
// //     ))
// //     {
// //         std::cerr<<"fail to settensorAddress input";
// //     }
// //     if(!trtengine.context_->setTensorAddress(
// //         trtengine.output_name[0].c_str(),
// //         device_output
// //     ))
// //     {
// //         std::cerr<<"fail to settensorAddress output";
// //     }

// //     bool success=trtengine.context_->enqueueV3(stream);

// //     if(!success){
// //         std::cerr<<"TensorRT enqueueV3 failed";
// //     }

// //     cuda_check(
// //         cudaMemcpyAsync(
// //             trtengine.host_output.data(),
// //             device_output,
// //             trtengine.output_datasize,
// //             cudaMemcpyDeviceToHost,
// //             stream
// //         )
// //     );

// //     cudaStreamSynchronize(stream);

// //     // size_t single_output_elements = 84 * 8400;
// //     // std::cout << trtengine.host_output[0] << '\n';
// //     // std::cout << trtengine.host_output[single_output_elements] << '\n';
// //     // std::cout << trtengine.host_output[2 * single_output_elements] << '\n';
// //     // std::cout << trtengine.host_output[3 * single_output_elements] << '\n';

// //     std::cout << "output[0]      = " << trtengine.host_output.at(0) << '\n';
// //     std::cout << "output[1]      = " << trtengine.host_output.at(1) << '\n';
// //     std::cout << "output[10]     = " << trtengine.host_output.at(10) << '\n';
// //     std::cout << "output[100]    = " << trtengine.host_output.at(100) << '\n';
// //     std::cout << "output[1000]   = " << trtengine.host_output.at(1000) << '\n';
// //     std::cout << "output[mid]    = " << trtengine.host_output.at(705600 / 2) << '\n';
// //     std::cout << "output[last]   = " << trtengine.host_output.at(705600 - 1) << '\n';

// //     // printTensorStats(trtengine.host_output);

// //     std::ofstream ofs(out_BIN_PATH,std::ios::binary);

// //     ofs.write(
// //         reinterpret_cast<const char*>(trtengine.host_output.data()), 
// //         trtengine.host_output.size() * sizeof(float)
// //     );

// // }