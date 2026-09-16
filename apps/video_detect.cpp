#include <iostream>
#include <vector>
#include<fstream>
#include <NvInfer.h>
#include<cstdint>
#include<cstddef>
#include <array>
#include <opencv2/opencv.hpp>
#include <thread>
#include<functional>
#include <atomic>
#include <nvtx3/nvToolsExt.h>

#include "trt_engine.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"
#include "cuda_preprocess.hpp"
#include "fram_slot.hpp"
#include "tools.hpp"
#include"cuda_postprocess.hpp"

#define video_path "assets/island.mp4"

#define model_PATH "models/yolov8n_cpp_fp16_profile.engine"

float confidence_threshold=0.20;
float iou_threshold=0.70;

//capture已经准备好，但gpu还没开始消费的工作
BoundedQueue<FramePacket> frame_queue(4);

//gpu已经处理完，但cpu postprocess还没消费的结果
BoundedQueue<ResultPacket> result_queue(2);

//保证同一个slot不会被两个frame同时使用
BoundedQueue<int> free_slots(2);

std::array<FramSlot,2>slots_;

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

int main(){

    std::atomic<int64_t> processed_frames{0};

    free_slots.pushBlocking(0);
    free_slots.pushBlocking(1);

    int dst_height=640;
    int dst_width=640;
    cv::VideoCapture probe(video_path);
    cv::Mat first_frame;
    if(!probe.isOpened() || !probe.read(first_frame) || first_frame.empty())
    {
        std::cerr<<"Failed to read first video frame\n";
        return -1;
    }
    int src_height=first_frame.rows;
    int src_width=first_frame.cols;
    probe.release();
    first_frame.release();

    size_t input_element=3ULL*dst_height*dst_width;
    size_t input_bytes=input_element*sizeof(float);

    trtengine trtengine;

    cv::VideoWriter writer(
        "result.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        30.0,
        cv::Size(src_width,src_height)
    );

    std::ofstream result_file("result.csv");
    if(!result_file.is_open())
    {
        std::cerr<<"Failed to open result.cv\n";

        return -1;
    }

    result_file<<"frame_id.class_id,confidence,x1,y1,x2,y2\n";

    trtengine.batch=1;
    nvinfer1::Dims4 dim{trtengine.batch,3,640,640};
    if(!trtengine.load(model_PATH))
        {
            std::cout<<"模型加载"<<std::endl;
            return -1;
        }

    if(!trtengine.createContext())
        {
            std::cout<<"创建context"<<std::endl;
            return -1;
        }

    if(!trtengine.getIOname())
        {
            return -1;
        }

    if(!trtengine.setInputShape(dim))
        {
            return -1;
        }

    if(!trtengine.setupIO())
        {
            return -1;
        }

    cudaStream_t preprocess_stream=nullptr;
    cuda_check(cudaStreamCreate(&preprocess_stream));
    postprocess pos{
        confidence_threshold,
        iou_threshold
    };
    cuda_check(cudaStreamCreate(&trtengine.stream));


//对slot初始化
    for(auto& slot:slots_)
    {
        //allocate gpu preprocess 的空间
        slot.image=
            cv::Mat(
                src_height,
                src_width,
                CV_8UC3
            );

        if(!slot.image.isContinuous())
        {
            std::cerr<<"slot image not continuous\n";
            return -1;
        }

        size_t src_bytes_0=slot.image.step*slot.image.rows;
        
        slot.image_capacity=src_bytes_0;

        cudaError_t err1=cudaHostRegister(
            slot.image.data,
            src_bytes_0,
            cudaHostRegisterMapped
        );

        if(err1!=cudaSuccess)
        {
            std::cerr<<"cudaHostRegister fail:"<<cudaGetErrorString(err1)<<std::endl;

            return -1;
        }

        cudaError_t err2=cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&slot.device_src),
            slot.image.data,
            0
        );

        if(err2!=cudaSuccess)
        {
            std::cerr<<"cudaHostGetDevicePointer"<<cudaGetErrorString(err2)<<std::endl;

        }
        //allcote preprocess的处理结果
        cuda_check(cudaMalloc(
            reinterpret_cast<void**>(&slot.device_input),
            input_bytes
        ));

        cuda_check(cudaMalloc(
            &slot.device_output,
            trtengine.output_bytes_
        ));

        cuda_check(cudaMallocHost(
            reinterpret_cast<void**>(&slot.host_output),
            trtengine.output_bytes_
        ));

        cuda_check(cudaMalloc(
            &slot.device_detections,
            8400*sizeof(detection)   
        ));

        cuda_check(cudaMalloc(
            &slot.device_det_count,
           sizeof(int)   
        ));

        cuda_check(cudaMallocHost(
            &slot.host_detections,
            8400*sizeof(detection)   
        ));

        cuda_check(cudaMallocHost(
            &slot.host_det_count,
            sizeof(int) 
        ));

        cudaEventCreateWithFlags(&slot.preprocess_done,cudaEventDisableTiming);
        cudaEventCreateWithFlags(&slot.output_ready,cudaEventDisableTiming);

        slot.meta.new_size=640;
        if(!slot.meta.computemeta(slot.image))
        {
            std::cout<<"meta计算失误";
        }
    }
// Warm up valid input buffers before measurement / graph capture.
    for(auto& slot:slots_)
    {
        cuda_check(cudaMemsetAsync(slot.device_input, 0, input_bytes, trtengine.stream));
        for(int i=0;i<10;i++)
        {
            if(!trtengine.inferfram(slot.device_input, slot.device_output))
                return -1;

        }
        cuda_check(cudaStreamSynchronize(trtengine.stream));

    }

    std::thread captureLoop_thread
    (
        captureLoop,
        video_path,
        std::ref(free_slots),
        std::ref(frame_queue),
        std::ref(slots_)
    );
    std::thread gpuloop_thread
    (
        gpuloop,
        std::ref(frame_queue),
        std::ref(result_queue),
        std::ref(free_slots),
        std::ref(slots_),
        confidence_threshold,
        dst_width,
        dst_height,
        preprocess_stream,
        std::ref(trtengine)
    );
    std::thread cpupostloop_thread
    (
        cpupostloop,
        std::ref(result_queue),
        std::ref(free_slots),
        std::ref(slots_),
        std::ref(pos),
        confidence_threshold,
        std::ref(result_file),
        std::ref(writer),
        std::ref(processed_frames)
    );

    auto start=Clock::now();
    captureLoop_thread.join();
    gpuloop_thread.join();
    cpupostloop_thread.join();
    auto end=Clock::now();

    double seconds=
        std::chrono::duration<double>(end-start).count();
    double fps=
        static_cast<double>(processed_frames.load())/seconds;
    
    std::cout
        <<"frames="
        <<processed_frames.load()
        <<'\n';

    std::cout
        <<"second="
        <<seconds
        <<'\n';

    std::cout
        <<"fpss="
        <<fps
        <<'\n';
    
}
