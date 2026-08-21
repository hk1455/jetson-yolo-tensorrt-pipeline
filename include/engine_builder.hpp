#include<iostream>


struct BuildOptions{
    std::string onnx_path;
    std::string engine_path;

    int min_batch=1;
    int opt_batch=4;
    int max_batch=8;

    bool fp16=false;

};


class EngineBuilder{
    public:
        bool build(const BuildOptions& options);
};


