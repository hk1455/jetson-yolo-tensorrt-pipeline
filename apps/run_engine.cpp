#include "../include/trt_engine.hpp"
#include <NvInfer.h>

const char* MODEL_PATH="models/yolov8n_cpp_fp32.engine";

const char* In_BIN_PATH="results/cpp/bus_input.bin";
const char* out_BIN_PATH="results/cpp/bus_raw_output.bin";

int main()
{
    trtengine trtengine;
    trtengine.load(MODEL_PATH);
    trtengine.printTensorInfo();

    trtengine.createContext();
    size_t in_size;
    int batch=1;
    trtengine.readFloatBinary(In_BIN_PATH,in_size);
    trtengine.infer(batch,out_BIN_PATH);
//     trtengine trtengine;
//     trtengine.load(MODEL_PATH);
//     trtengine.printTensorInfo();

//     trtengine.createContext();
//     trtengine.setInputShape(nvinfer1::Dims4 {3,3,640,640});
//     trtengine.printRuntime();


}