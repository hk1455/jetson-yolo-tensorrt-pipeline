#include <iostream>
#include <vector>
#include<fstream>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

#include "engine_builder.hpp"
#include "trt_engine.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"

#define MODEL_PATH "models/yolov8n_cpp_fp32.engine"
#define IMAGE_PATH "assets/home.jpeg"
#define OUT_PUT "results/benchmark/home_detection.jpeg"

float confidence_threshold=0.20;
float iou_threshold=0.70;

using Clock=std::chrono::steady_clock;



double mean(const std::vector<double>& values){
    double sum=0.0;

    for(double v:values)
    {
        sum+=v;
    }
    return sum/values.size();
};

double findp(std::vector<double> data,double p)
{
    if(data.empty())
        return 0.0;
    size_t n=data.size();
    double rank=p*(n-1);
    size_t lo = static_cast<size_t>(std::floor(rank));
    size_t hi = static_cast<size_t>(std::ceil(rank));

    std::nth_element(data.begin(),data.begin()+lo,data.end());
    double lon=data[lo];

    if(lo==hi)
        return lon;

    std::nth_element(data.begin(),data.begin()+hi,data.end());
    double hin=data[hi];

    return hin-(hin-lon)*(hi-rank);
   
};

double findmin(std::vector<double> data)
{
    if(data.empty())
        return 0.0;

    std::nth_element(data.begin(),data.begin()+1,data.end());
    double min=data[0];

    return min;
};

double findmax(std::vector<double> data)
{
    if(data.empty())
        return 0.0;

    std::nth_element(data.begin(),data.end()-1,data.end());
    double max=data[199];

    return max;
};

int main(){
    std::ofstream csv("day16_memory_fp32_b1.csv");
    csv<<"run_ID,Preprocess,Staging,Host Infer,Core E2E,NMS,PROCESS,REVERSE,H2D,TRT,D2H\n";
    float* pinned_input=nullptr;
    float* pinned_output=nullptr;

    std::vector<float> input;

//前处理
    letterboxmeta meta{
        0,
        0,
        0,
        0,
        640,
        0,
        0,
        0,
    };

    cv::Mat image1;
    image1=cv::imread(IMAGE_PATH,cv::IMREAD_COLOR);

    trtengine trtengine;
    if(!trtengine.load(MODEL_PATH))
    {
        std::cout<<"模型加载"<<std::endl;
        return -1;
    }

    if(!trtengine.createContext())
    {
        std::cout<<"创建context"<<std::endl;
        return -1;
    }
    trtengine.batch=1;
    nvinfer1::Dims4 dim{trtengine.batch,3,640,640};

    if(!trtengine.prepare(dim))
    {
        std::cout<<"prepare出错"<<std::endl;
        return -1;        
    }

    cudaMallocHost(
        reinterpret_cast<void**>(&pinned_input),
        trtengine.input_bytes_);

    cudaMallocHost(
        reinterpret_cast<void**>(&pinned_output),
        trtengine.output_bytes_);

    postprocess pos{
        confidence_threshold,
        iou_threshold
    };

    RuntimeTiming timing;

    for(int i=0;i<100;i++)
    {
          if(!preprocess(image1,input,meta))
        {
            std::cout<<"前处理错误"<<std::endl;
            return -1;
        }

        std::memcpy(
            pinned_input,
            input.data(),
            trtengine.input_bytes_
        );
    //runengine

        if(!trtengine.infer(pinned_input,pinned_output,&timing))
        {
            std::cout<<"推理出错"<<std::endl;
            return -1;
        }

    //postprocess
        std::vector<detection> det=pos.process(pinned_output,meta,confidence_threshold);

        std::vector<detection> det3=pos.nms(det);

        for(int i=0;i<det3.size();i++)
        { 
            pos.size2original(
                det3[i],
                meta.scale,
                meta.pad_x,
                meta.pad_y,
                meta.original_width,
                meta.original_height
            );
        }
    }

    constexpr int RUNS=200;
    std::vector<double> preprocess_times;
    std::vector<double> staging_times;

    std::vector<double> infer_times;    
    std::vector<double> postprocess_times;
    std::vector<double> e2e_times;
    std::vector<double> nms_times;
    std::vector<double> process_times;
    std::vector<double> reverse_times;

    preprocess_times.reserve(RUNS);
    staging_times.reserve(RUNS);
    infer_times.reserve(RUNS);
    postprocess_times.reserve(RUNS);
    e2e_times.reserve(RUNS);
    nms_times.reserve(RUNS);
    process_times.reserve(RUNS);
    reverse_times.reserve(RUNS);

    RuntimeTiming timing1;
    std::vector<double> h2d;
    std::vector<double>trt;
    std::vector<double>d2h;   
    
    h2d.reserve(RUNS);
    trt.reserve(RUNS);
    d2h.reserve(RUNS);


    for(int i=0;i<RUNS;i++)
    {
        //开始计时
        auto t0=Clock::now();

        if(!preprocess(image1,input,meta))
        {
            std::cout<<"前处理错误"<<std::endl;
            return -1;
        }

        auto t6=Clock::now();

        std::memcpy(
            pinned_input,
            input.data(),
            trtengine.input_bytes_
        );

        auto t1=Clock::now();

    //runengine

        if(!trtengine.infer(pinned_input,pinned_output,&timing1))
        {
            std::cout<<"推理出错"<<std::endl;
            return -1;
        }
        auto t2=Clock::now();

    //postprocess
        std::vector<detection> det=pos.process(pinned_output,meta,confidence_threshold);
        auto t3=Clock::now();

        std::vector<detection> det3=pos.nms(det);
        auto t4=Clock::now();

        for(int i=0;i<det3.size();i++)
        { 
            pos.size2original(
                det3[i],
                meta.scale,
                meta.pad_x,
                meta.pad_y,
                meta.original_width,
                meta.original_height
            );
        }
        auto t5=Clock::now();

        double preprocess_ms =
            std::chrono::duration<double, std::milli>(
                t6 - t0
            ).count();

        double staging_ms =
                std::chrono::duration<double, std::milli>(
                    t1 - t6
            ).count();   

        double infer_ms =
                std::chrono::duration<double, std::milli>(
                    t2 - t1
            ).count();
        double process_ms =
                std::chrono::duration<double, std::milli>(
                    t3 - t2
            ).count();
        double nms_ms =
                std::chrono::duration<double, std::milli>(
                    t4 - t3
            ).count();
        double reverse_ms =
                std::chrono::duration<double, std::milli>(
                    t5 - t4
            ).count();
        double postprocess_ms =
                std::chrono::duration<double, std::milli>(
                    t5 - t2
            ).count();
        double e2e_ms =
                std::chrono::duration<double, std::milli>(
                    t5 - t0
            ).count();

        preprocess_times.push_back(preprocess_ms);
        staging_times.push_back(staging_ms);
        infer_times.push_back(infer_ms);
        postprocess_times.push_back(postprocess_ms);
        e2e_times.push_back(e2e_ms);
        nms_times.push_back(nms_ms);
        process_times.push_back(process_ms);
        reverse_times.push_back(reverse_ms);


        h2d.push_back(timing1.h2d_ms);
        trt.push_back(timing1.trt_ms);
        d2h.push_back(timing1.d2h_ms);
    }

    
    std::cout
        <<"mean:  "<<"ms\n"
        << "Preprocess   : " <<mean(preprocess_times) << " ms\n"
        << "staging memory: " <<mean(staging_times) << " ms\n"
        << "Host Infer   : " <<mean(infer_times)<< " ms\n"
        << "Postprocess  : " << mean(postprocess_times) << " ms\n"
        << "Core E2E     : " <<mean(e2e_times) << " ms\n"
        << "NMS          : " <<mean(nms_times) << " ms\n"
        << "PROCESS      : " <<mean(process_times) << " ms\n"
        << "REVERSE      : " <<mean(reverse_times) << " ms\n";

    std::cout
        <<"p50:  "<<"ms\n"
        << "Preprocess   : " <<findp(preprocess_times,0.5) << " ms\n"
        << "staging memory: " <<findp(staging_times,0.5) << " ms\n"
        << "Host Infer   : " <<findp(infer_times,0.5)<< " ms\n"
        << "Postprocess  : " << findp(postprocess_times,0.5) << " ms\n"
        << "Core E2E     : " <<findp(e2e_times,0.5) << " ms\n"
        << "NMS          : " <<findp(nms_times,0.5) << " ms\n"
        << "PROCESS      : " <<findp(process_times,0.5) << " ms\n"
        << "REVERSE      : " <<findp(reverse_times,0.5) << " ms\n";

    std::cout
        <<"p95:  "<<"ms\n"
        << "Preprocess   : " <<findp(preprocess_times,0.95) << " ms\n"
        << "staging memory: " <<findp(staging_times,0.95) << " ms\n"
        << "Host Infer   : " <<findp(infer_times,0.95)<< " ms\n"
        << "Postprocess  : " << findp(postprocess_times,0.95) << " ms\n"
        << "Core E2E     : " <<findp(e2e_times,0.95) << " ms\n"
        << "NMS          : " <<findp(nms_times,0.95) << " ms\n"
        << "PROCESS      : " <<findp(process_times,0.95) << " ms\n"
        << "REVERSE      : " <<findp(reverse_times,0.95) << " ms\n";
    std::cout
        <<"min:  "<<"ms\n"
        << "Preprocess   : " <<findmin(preprocess_times) << " ms\n"
        << "staging memory: " <<findmin(staging_times) << " ms\n"
        << "Host Infer   : " <<findmin(infer_times)<< " ms\n"
        << "Postprocess  : " << findmin(postprocess_times) << " ms\n"
        << "Core E2E     : " <<findmin(e2e_times) << " ms\n"
        << "NMS     : " <<findmin(nms_times) << " ms\n"
        << "PROCESS     : " <<findmin(process_times) << " ms\n"
        << "REVERSE     : " <<findmin(reverse_times) << " ms\n";
    std::cout
        <<"max:  "<<"ms\n"
        << "Preprocess   : " <<findmax(preprocess_times) << " ms\n"
        << "staging memory: " <<findmax(staging_times) << " ms\n"
        << "Host Infer   : " <<findmax(infer_times)<< " ms\n"
        << "Postprocess  : " << findmax(postprocess_times) << " ms\n"
        << "Core E2E     : " <<findmax(e2e_times) << " ms\n"
        << "NMS          : " <<findmax(nms_times) << " ms\n"
        << "PROCESS      : " <<findmax(process_times) << " ms\n"
        << "REVERSE      : " <<findmax(reverse_times) << " ms\n";

    std::cout
        <<"mean:  "<<"ms\n"
        <<"H2D:"<<mean(h2d)<< " ms\n"
        <<"TRT:"<<mean(trt)<< " ms\n"
        <<"D2H:"<<mean(d2h)<< " ms\n";
    std::cout
        <<"p50:  "<<"ms\n"
        <<"H2D:"<<findp(h2d,0.5)<< " ms\n"
        <<"TRT:"<<findp(trt,0.5)<< " ms\n"
        <<"D2H:"<<findp(d2h,0.5)<< " ms\n";
    std::cout
        <<"p95:  "<<"ms\n"
        <<"H2D:"<<findp(h2d,0.95)<< " ms\n"
        <<"TRT:"<<findp(trt,0.95)<< " ms\n"
        <<"D2H:"<<findp(d2h,0.95)<< " ms\n";
    std::cout
        <<"min:  "<<"ms\n"
        <<"H2D:"<<findmin(h2d)<< " ms\n"
        <<"TRT:"<<findmin(trt)<< " ms\n"
        <<"D2H:"<<findmin(d2h)<< " ms\n";
    std::cout
        <<"max:  "<<"ms\n"
        <<"H2D:"<<findmax(h2d)<< " ms\n"
        <<"TRT:"<<findmax(trt)<< " ms\n"
        <<"D2H:"<<findmax(d2h)<< " ms\n";


    for(size_t i=0;i<preprocess_times.size();++i)
    {
        csv<<i<<","
        <<preprocess_times[i]<<","
        <<staging_times[i]<<","
        <<infer_times[i]<<","
        <<postprocess_times[i]<<","
        <<e2e_times[i]<<","
        <<nms_times[i]<<","
        <<process_times[i]<<","
        <<reverse_times[i]<<","
        <<h2d[i]<<","
        <<trt[i]<<","
        <<d2h[i]<<","
        <<"\n";
    }


    cudaFreeHost(pinned_input);
    cudaFreeHost(pinned_output);
}