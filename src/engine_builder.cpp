#pragma once

#include<string>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include<fstream>

#include "../include/engine_builder.hpp"

class Logger:public nvinfer1::ILogger{
    public:
        void log(
            Severity severity,
            const char* msg
            )noexcept override
        {
            if(severity<=Severity::kWARNING)
                {
                    std::cout<<"{TensorRT}"<<msg<<std::endl;
                }
        }
};


bool EngineBuilder::build(const BuildOptions& options){   
    const char* MODER_PATH=options.onnx_path.c_str();

    Logger logger;

    auto* builder=nvinfer1::createInferBuilder(logger);

    auto* network=builder->createNetworkV2(0U);

    auto* config=builder->createBuilderConfig();

    auto* profile=builder->createOptimizationProfile();

    auto* parser=nvonnxparser::createParser(
        *network,
        logger
    );
    bool parsed=parser->parseFromFile(
        MODER_PATH,
        static_cast<int>(
            nvinfer1::ILogger::Severity::kWARNING
        )
    );

    if(!parsed){
        std::cerr<<"Failed to parse OMMX\n" ;
        return 0;
    }

    auto* input_tensor=network->getInput(0);
    auto* input_name=input_tensor->getName();

    profile->setDimensions(
        input_name,
        nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{options.min_batch,3,640,640}
    );
    profile->setDimensions(
        input_name,
        nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{options.opt_batch,3,640,640}
    );
    profile->setDimensions(
        input_name,
        nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{options.max_batch,3,640,640}
    );
    
    config->addOptimizationProfile(profile);

    config->clearFlag(nvinfer1::BuilderFlag::kTF32);

    if(options.fp16){
        config->setFlag(
            nvinfer1::BuilderFlag::kFP16
        );
    }

    auto* serialized_engine=builder->buildSerializedNetwork(
        *network,
        *config
    );

    if(!serialized_engine){
        std::cerr<<"Failed to build engine\n" ;
        return 0;       
    }

    std::ofstream file(options.engine_path,std::ios::binary);

    const char* data=static_cast<const char*>(serialized_engine->data());
    file.write(
        data,
        serialized_engine->size()
    );

    return 1;
};