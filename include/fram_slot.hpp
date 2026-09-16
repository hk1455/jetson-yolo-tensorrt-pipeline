#pragma once

#include <array>
#include <chrono>
#include <mutex>
#include <fstream>
#include <cstdlib>
#include<cstdint>
#include<cstddef>
#include <condition_variable>
#include<algorithm>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <nvtx3/nvToolsExt.h>
#include <NvInfer.h>
#include <deque>

#include"cuda_preprocess.hpp"
#include"cuda_postprocess.hpp"
#include "postprocess.hpp"
#include "trt_engine.hpp"

using Clock =std::chrono::steady_clock;

class NvtxRange{
public:
    explicit NvtxRange(const char* name){
        nvtxRangePushA(name);
    }

    ~NvtxRange(){
        nvtxRangePop();
    }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

struct FramSlot
{
    int frame_id=-1;
    cv::Mat image;
//gpu视角图像的原始输入
    uint8_t* device_src=nullptr;
//图像经过preprocess，作为tensorrt的输入
    float* device_input=nullptr;
    float* device_output=nullptr;
    float* host_output=nullptr;

    detection* device_detections=nullptr;
    int* device_det_count=nullptr;

    detection* host_detections=nullptr;
    int* host_det_count=nullptr;

    letterboxmeta meta;
    size_t image_capacity=0;

    cudaEvent_t preprocess_done;
    cudaEvent_t inference_done;
    cudaEvent_t output_ready;

    cudaGraph_t graph=nullptr;
    cudaGraphExec_t graph_exec=nullptr;

    Clock::time_point capture_time;
    Clock::time_point gpu_submit_time;
    Clock::time_point post_start_time;
    Clock::time_point detection_done_time;
};

struct FramePacket
{
    int slot_index=-1;
};

struct ResultPacket
{
    int64_t slot_index=-1;
};

template <typename T>
class BoundedQueue
{
    public:
        explicit BoundedQueue(size_t capacity)
            : capacity_(capacity)
        {
        }

        bool pushBlocking(T item)
        {
            std::unique_lock<std::mutex> lock(mutex_);

            not_full_.wait(
                lock,
                [&]()
                {
                    return queue_.size() < capacity_  
                        || closed_; 
                }
            );

            if(closed_)
                return false;
            
            queue_.push_back(std::move(item));

            not_empty_.notify_one();

            return true;
        }

        bool pop(T& item)
        {
            std::unique_lock<std::mutex>lock(mutex_);
            not_empty_.wait(
                lock,
                [&]()
                {
                    return !queue_.empty()
                        ||closed_;
                }
            );

            if(queue_.empty())
                return false;
            
            item =std::move(queue_.front());

            queue_.pop_front();

            not_full_.notify_one();

            return true;
        }

        void close()
        {
            {
                std::lock_guard<std::mutex>lock(mutex_);
                closed_=true;
            }

            not_empty_.notify_all();
            not_full_.notify_all();
        }


    private:
        std::deque<T> queue_;

        size_t capacity_;

        std::mutex mutex_;

        std::condition_variable not_empty_;
        std::condition_variable not_full_;

        bool closed_=false;
};



void captureLoop(
    const std::string& video_path,
    BoundedQueue<int>& free_slot,
    BoundedQueue<FramePacket>& frame_queue,
    std::array<FramSlot,2>& slots
);

void gpuloop
(
    BoundedQueue<FramePacket>& frame_queue,
    BoundedQueue<ResultPacket>& result_packet,
    BoundedQueue<int>& free_slot,
    std::array<FramSlot, 2>& slots,
    float confidence_threshold,
    int dst_width,
    int dst_height,
    cudaStream_t preprocess_stream,
    trtengine& trtengine
);

void cpupostloop
(
    BoundedQueue<ResultPacket>& result_packet,
    BoundedQueue<int>& free_slot,
    std::array<FramSlot, 2>& slots,
    postprocess& pos,
    float confidence_threshold,
    std::ofstream& result_file,
    cv::VideoWriter& writer,
    std::atomic<int64_t>& processed_frames
);


bool initcudagraph(
    FramSlot& slot,
    trtengine& trtengine,
    float confidence_threshold
);
