#include<iostream>
#include "../include/engine_builder.hpp"

int main(){
    const BuildOptions options{
            "models/yolov8n.onnx",
            "models/yolov8n_cpp_fp16_profile.engine",//这里是tf32的engine
            1,
            4,
            8,
            true
        };
    EngineBuilder builder;

    bool success=builder.build(options);

    return 0;
}