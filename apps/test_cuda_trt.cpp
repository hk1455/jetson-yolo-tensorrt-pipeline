#include <vector>
#include<fstream>
#include <iostream>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cmath>

#include "engine_builder.hpp"
#include "trt_engine.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"
#include "cuda_preprocess.hpp"

#define MODEL_PATH "models/yolov8n_cpp_fp16.engine"
#define IMAGE_PATH "assets/home.jpeg"
#define OUT_PATH "results/cuda_preprocess/home_raw_output_V6_fp16.bin"


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

    std::nth_element(data.begin(),data.begin(),data.end());
    double min=data[0];

    return min;
};

double findmax(std::vector<double> data)
{
    if(data.empty())
        return 0.0;

    std::nth_element(data.begin(),data.end()-1,data.end());
    double max=data[4999];

    return max;
};

int main()
{
    RuntimeTiming timing;
    const int dst_width=640;
    const int dst_height=640;

    cv::Mat image=cv::imread(IMAGE_PATH);

    if(image.empty())
    {
        std::cerr<<"fail to load image"<<std::endl;
        return -1;
    }

    if(!image.isContinuous())
    {
        image=image.clone();
    }

//cpu preprocess,为了得到meta
    letterboxmeta meta;
    meta.new_size=640;

    std::vector<float> cpu_output;

    if(!meta.computemeta(image))
    {
        std::cout<<"meta计算失误";
    }

// allocate GPU source


    uint8_t* device_src=nullptr;
    size_t src_bytes=
        image.step*image.rows;
    
    cudaError_t err=cudaHostRegister(
        image.data,
        src_bytes,
        cudaHostRegisterDefault
    );

    if(err!=cudaSuccess)
    {
        std::cerr<<"cudaHostRegister fail:"<<cudaGetErrorString(err)<<std::endl;

        return -1;
    }
    cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&device_src),
        image.data,
        0
    );
    // cudaMalloc(
    //     reinterpret_cast<void**>(&device_src),
    //     src_bytes
    // );

    uint8_t* pinned_src=nullptr;
    cudaMallocHost(
        reinterpret_cast<void**>(&pinned_src),
        src_bytes
    );
    std::memcpy(
        pinned_src,
        image.data,
        src_bytes
    );

    cudaMemcpyAsync(
        device_src,
        pinned_src,
        src_bytes,
        cudaMemcpyHostToDevice
    );
    //allcote GPU output

    size_t input_element=3ULL*dst_height*dst_width;

    size_t input_bytes=input_element*sizeof(float);
//preprocess的处理结果
    float* device_dst=nullptr;

    cudaMalloc(
        reinterpret_cast<void**>(&device_dst),
        input_bytes
    );

    trtengine trtengine;

    trtengine.batch=1;
    nvinfer1::Dims4 dim{trtengine.batch,3,640,640};
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

    cudaMalloc(
        &trtengine.device_output,
        trtengine.output_bytes_
    );
    cudaEvent_t e_;
    cudaEventCreate(&e_);

    cudaEventCreate(&trtengine.e0_);
    cudaEventCreate(&trtengine.e1_);
    cudaEventCreate(&trtengine.e2_);
    cudaEventCreate(&trtengine.e3_);

    constexpr int RUNS=5000;
    cudaStreamCreate(&trtengine.stream);
    
    float* host_output=nullptr;
    cudaMallocHost(
        reinterpret_cast<void**>(&host_output),
        trtengine.output_bytes_
    );

    for(int i=0;i<10;i++)
    {    
    
        cudaMemcpyAsync(
            device_src,
            image.data,
            src_bytes,
            cudaMemcpyHostToDevice,
            trtengine.stream
        );
        bool ok=cudaPreprocessV1(
                device_src,
                image.cols,
                image.rows,
                image.step,

                device_dst,
                dst_width,
                dst_height,
                
                meta,
                trtengine.stream
            );

        if(!ok)
            {
                std::cerr<<"fail to CUDA preprocess\n";
                return -1;
            }

    
        if(!trtengine.inferDevice(
            device_dst,
            &timing      
        )
            )
            {
                std::cout<<"inferDevice"<<std::endl;
                return -1;
            }

        cudaMemcpyAsync(
            host_output,
            trtengine.device_output,
            trtengine.output_bytes_,
            cudaMemcpyDeviceToHost,
            trtengine.stream
        );

        cudaStreamSynchronize(trtengine.stream);
    }

//将raw_output写入文件
    std::ofstream outfile(OUT_PATH,std::ios::binary);
    if(!outfile)
        std::cout<<"打开输出路径错误"<<std::endl;    

    outfile.write(
        reinterpret_cast<char*>(host_output),
        trtengine.output_bytes_
    );


    std::vector<double>memcpy_times;
    std::vector<double>raw_h2d_times;
    std::vector<double> preprocess_times;
    std::vector<double> infer_times;    
    std::vector<double> d2h_times;
    std::vector<double>total_times;



    std::vector<double>cudaPreprocessV1CPUcall_times; 
    std::vector<double>event1recordtimes;            
    std::vector<double>inferDeviceCPUcall_times;      
    std::vector<double>event2recordtimes;            
    std::vector<double>D2HasyncAPIcall_times ;       


    float cpu2gpu_ms=0.0f;
    float gpu_ms=0.0f;

    float memcpy_ms=0.0f;
    float raw_h2d_ms=0.0f;
    float preprocess_ms=0.0f;
    float infer_ms=0.0f;  
    float d2h_ms=0.0f;
    float total_ms=0.0f;

    cudaPreprocessV1CPUcall_times.reserve(RUNS);
    event1recordtimes.reserve(RUNS);
    inferDeviceCPUcall_times.reserve(RUNS);      
    event2recordtimes.reserve(RUNS);            
    D2HasyncAPIcall_times.reserve(RUNS);   

    memcpy_times.reserve(RUNS);
    raw_h2d_times.reserve(RUNS);
    preprocess_times.reserve(RUNS);
    infer_times.reserve(RUNS);
    d2h_times.reserve(RUNS);
    total_times.reserve(RUNS);

    for(int i=0;i<RUNS;i++)
    {   
        //开始计时
        cudaEventRecord(trtengine.e0_,trtengine.stream);

        auto t0=Clock::now();

        bool ok=cudaPreprocessV1(
                device_src,
                image.cols,
                image.rows,
                image.step,

                device_dst,
                dst_width,
                dst_height,
                
                meta,
                trtengine.stream
            );

        auto t1=Clock::now();

        if(!ok)
            {
                std::cerr<<"fail to CUDA preprocess\n";
                return -1;
            }

        cudaEventRecord(trtengine.e1_,trtengine.stream);

        auto t2=Clock::now();
        if(!trtengine.inferDevice(
            device_dst,
            &timing
        )
            )
            {
                std::cout<<"inferDevice"<<std::endl;
                return -1;
            }


        auto t3=Clock::now();

        cudaEventRecord(trtengine.e2_,trtengine.stream);

        auto t4=Clock::now();

        cudaMemcpyAsync(
            host_output,
            trtengine.device_output,
            trtengine.output_bytes_,
            cudaMemcpyDeviceToHost,
            trtengine.stream
        );

        auto t5=Clock::now();

        cudaEventRecord(trtengine.e3_,trtengine.stream);

        cudaEventSynchronize(trtengine.e3_);
        auto t6=Clock::now();

        cpu2gpu_ms=std::chrono::duration<float,std::milli>(
            t1-t0
        ).count();    

        gpu_ms=std::chrono::duration<float,std::milli>(
            t2-t1
        ).count();    

        float inferDeviceCPUcall_ms=std::chrono::duration<float,std::milli>(
            t3 - t2
        ).count();   

        float event2record_ms=std::chrono::duration<float,std::milli>(
            t4 - t3
        ).count();   

        float  D2HasyncAPIcall_ms=std::chrono::duration<float,std::milli>(
            t5 - t4
        ).count();   

        cudaEventElapsedTime(
            &preprocess_ms,
            trtengine.e0_,
            trtengine.e1_
        );
        cudaEventElapsedTime(
            &infer_ms,
            trtengine.e1_,
            trtengine.e2_
        );
        cudaEventElapsedTime(
            &d2h_ms,
            trtengine.e2_,
            trtengine.e3_
        );
        cudaEventElapsedTime(
            &total_ms,
            trtengine.e0_,
            trtengine.e3_
        );

        // memcpy_times.push_back(memcpy_ms);
        cudaPreprocessV1CPUcall_times.push_back(cpu2gpu_ms);
        event1recordtimes.push_back(gpu_ms);

        inferDeviceCPUcall_times.push_back(inferDeviceCPUcall_ms);      
        event2recordtimes.push_back(event2record_ms);            
        D2HasyncAPIcall_times.push_back(D2HasyncAPIcall_ms);  

        raw_h2d_times.push_back(raw_h2d_ms);
        preprocess_times.push_back(preprocess_ms);
        infer_times.push_back(infer_ms);
        d2h_times.push_back(d2h_ms);
        total_times.push_back(total_ms);
    }

    std::cout
        <<"cudaPreprocessV1CPUcall_times: "
        << "mean: " <<mean(cudaPreprocessV1CPUcall_times) << " ms\n";
    std::cout
        <<"p50:  "<<findp(cudaPreprocessV1CPUcall_times,0.5) << " ms\n";
    std::cout
        <<"p95:  "<<findp(cudaPreprocessV1CPUcall_times,0.95) << " ms\n";
    std::cout
        <<"min:  "<<findmin(cudaPreprocessV1CPUcall_times) << " ms\n";
    std::cout
        <<"max:  "<<findmax(cudaPreprocessV1CPUcall_times) << " ms\n";

    std::cout
        <<"event1recordtimes: "
        << "mean: " <<mean(event1recordtimes) << " ms\n";
    std::cout
        <<"p50:  "<<findp(event1recordtimes,0.5) << " ms\n";
    std::cout
        <<"p95:  "<<findp(event1recordtimes,0.95) << " ms\n";
    std::cout
        <<"min:  "<<findmin(event1recordtimes) << " ms\n";
    std::cout
        <<"max:  "<<findmax(event1recordtimes) << " ms\n";

    std::cout
        <<"inferDeviceCPUcall_times: "
        << "mean: " <<mean(inferDeviceCPUcall_times) << " ms\n";
    std::cout
        <<"p50:  "<<findp(inferDeviceCPUcall_times,0.5) << " ms\n";
    std::cout
        <<"p95:  "<<findp(inferDeviceCPUcall_times,0.95) << " ms\n";
    std::cout
        <<"min:  "<<findmin(inferDeviceCPUcall_times) << " ms\n";
    std::cout
        <<"max:  "<<findmax(inferDeviceCPUcall_times) << " ms\n";

    std::cout
        <<"event2recordtimes: "
        << "mean: " <<mean(event2recordtimes) << " ms\n";
    std::cout
        <<"p50:  "<<findp(event2recordtimes,0.5) << " ms\n";
    std::cout
        <<"p95:  "<<findp(event2recordtimes,0.95) << " ms\n";
    std::cout
        <<"min:  "<<findmin(event2recordtimes) << " ms\n";
    std::cout
        <<"max:  "<<findmax(event2recordtimes) << " ms\n";

        
    std::cout
        <<" D2HasyncAPIcall_times: "
        << "mean: " <<mean( D2HasyncAPIcall_times) << " ms\n";
    std::cout
        <<"p50:  "<<findp( D2HasyncAPIcall_times,0.5) << " ms\n";
    std::cout
        <<"p95:  "<<findp( D2HasyncAPIcall_times,0.95) << " ms\n";
    std::cout
        <<"min:  "<<findmin( D2HasyncAPIcall_times) << " ms\n";
    std::cout
        <<"max:  "<<findmax( D2HasyncAPIcall_times) << " ms\n";



    std::cout
        <<"raw_h2d_times: "
        << "mean: " <<mean(raw_h2d_times) << " ms\n";
    std::cout
        <<"p50:  "<<findp(raw_h2d_times,0.5) << " ms\n";

    std::cout
        <<"p95:  "<<findp(raw_h2d_times,0.95) << " ms\n";
    std::cout
        <<"min:  "<<findmin(raw_h2d_times) << " ms\n";
    std::cout
        <<"max:  "<<findmax(raw_h2d_times) << " ms\n";

    std::cout
        <<"preprocess_times: "
        << "mean: " <<mean(preprocess_times) << " ms\n";
    std::cout
        <<"p50:  "<<findp(preprocess_times,0.5) << " ms\n";
    std::cout
        <<"p95:  "<<findp(preprocess_times,0.95) << " ms\n";
    std::cout
        <<"min:  "<<findmin(preprocess_times) << " ms\n";
    std::cout
        <<"max:  "<<findmax(preprocess_times) << " ms\n";

    std::cout
        <<"infer_times: "
        << "mean:" <<mean(infer_times) << " ms\n";
    std::cout
        <<"p50: "<<findp(infer_times,0.5) << " ms\n";
    std::cout
        <<"p95: " <<findp(infer_times,0.95) << " ms\n";
    std::cout
        <<"min: "<<findmin(infer_times) << " ms\n";
    std::cout
        <<"max: " <<findmax(infer_times) << " ms\n";

    std::cout
        <<"d2h_times: "
        << "mean:" <<mean(d2h_times) << " ms\n";
    std::cout
        <<"p50: "<<findp(d2h_times,0.5) << " ms\n";
    std::cout
        <<"p95: " <<findp(d2h_times,0.95) << " ms\n";
    std::cout
        <<"min: "<<findmin(d2h_times) << " ms\n";
    std::cout
        <<"max: " <<findmax(d2h_times) << " ms\n";

    std::cout
        <<"total_times: "
        << "mean:" <<mean(total_times) << " ms\n";
    std::cout
        <<"p50: "<<findp(total_times,0.5) << " ms\n";
    std::cout
        <<"p95: " <<findp(total_times,0.95) << " ms\n";
    std::cout
        <<"min: "<<findmin(total_times) << " ms\n";
    std::cout
        <<"max: " <<findmax(total_times) << " ms\n";        

}