#include "../include/trt_engine.hpp"
#include <NvInfer.h>

const char* MODEL_PATH="models/yolov8n_cpp_fp16.engine";


int main(){
    trtengine trtengine;
    trtengine.load(MODEL_PATH);
    trtengine.printTensorInfo();
    trtengine.createContext();
    trtengine.setInputShape(nvinfer1::Dims4 {1,3,640,640});
    trtengine.printRuntime();
}