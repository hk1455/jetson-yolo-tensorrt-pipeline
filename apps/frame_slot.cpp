#include <iostream>
#include <vector>
#include<fstream>
#include <NvInfer.h>
#include <array>
#include <opencv2/opencv.hpp>

#include "trt_engine.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"
#include "cuda_preprocess.hpp"
#include "fram_slot.hpp"
#include "tools.hpp"

#define image1_PATH "assets/home.jpeg"
#define image2_PATH "assets/bus.jpg"

#define model_PATH "models/yolov8n_cpp_fp16_profile.engine"

using Clock=std::chrono::steady_clock;

float confidence_threshold=0.20;
float iou_threshold=0.70;

std::array<FramSlot,2>slots_;

void process_frames(
    int RUNS,
    trtengine& trtengine,
    postprocess& pos,
    cudaStream_t preprocess_stream,
    int dst_width,
    int dst_height,
    std::vector<double>& postprocess_times,      
    std::vector<double>& nms_times,            
    std::vector<double>& size2original_times,
    std::vector<double>& all_times,
    
    std::vector<double>& infer_times,
    std::vector<double>& output_ready_cpuwait_times
    )
{
        FramSlot& slot0=slots_[0];

        bool ok=cudaPreprocessV1(
            slot0.device_src,
            slot0.image.cols,
            slot0.image.rows,
            slot0.image.step,

            slot0.device_input,
            dst_width,
            dst_height,
            
            slot0.meta,
            preprocess_stream
        );
                
    if(!ok)
    {
            std::cerr<<"fail to CUDA preprocess\n";
            return;
    }

    cudaEventRecord(slot0.preprocess_done,preprocess_stream);

    for(int frame_id=0;frame_id<RUNS;frame_id++)
    {
        FramSlot& current=slots_[frame_id%2];
        current.frame_id=frame_id;

        cudaStreamWaitEvent(
            trtengine.stream,
            current.preprocess_done,
            0
        );

        // auto t11=std::chrono::steady_clock::now();
        if(!trtengine.inferfram
            (
                current.device_input,
                current.device_output
            ))
            {
                std::cout<<"inferDevice"<<std::endl;
                return;
            }
        // auto t12=std::chrono::steady_clock::now();

//把结果转移到主机

        cudaMemcpyAsync(
            current.host_output,
            current.device_output,
            trtengine.output_bytes_,
            cudaMemcpyDeviceToHost,
            trtengine.stream
        );

        if(frame_id+1<RUNS)
        {
            FramSlot& next=slots_[(frame_id+1)%2];
            next.frame_id=frame_id+1;
            bool ok=cudaPreprocessV1(
                    next.device_src,
                    next.image.cols,
                    next.image.rows,
                    next.image.step,

                    next.device_input,
                    dst_width,
                    dst_height,
                    
                    next.meta,
                    preprocess_stream
                );
            
            if(!ok)
            {
                    std::cerr<<"fail to CUDA preprocess\n";
                    return;
            }

            cudaEventRecord(
                next.preprocess_done,
                preprocess_stream
            );
        }

        cudaEventRecord(current.output_ready,trtengine.stream);

        // auto t9=std::chrono::steady_clock::now();
        cudaEventSynchronize(current.output_ready);
        // auto t10=std::chrono::steady_clock::now();
//process
        // auto t4=std::chrono::steady_clock::now();
        std::vector<detection> det=pos.process(current.host_output,current.meta,confidence_threshold);
        // auto t5=std::chrono::steady_clock::now();

        const std::vector<detection> det2=det;

        // auto t6=std::chrono::steady_clock::now();
        std::vector<detection> det3=pos.nms(det2);
        // auto t7=std::chrono::steady_clock::now();

        for(int i=0;i<det3.size();i++)
            { 
                pos.size2original(det3[i],current.meta.scale,current.meta.pad_x,current.meta.pad_y,current.meta.original_width,current.meta.original_height);
            }

        // auto t8=std::chrono::steady_clock::now();


        // double process_ms=std::chrono::duration<double,std::milli>(
        //     t5-t4
        // ).count();

        // double nms_ms=std::chrono::duration<double,std::milli>(
        //     t7-t6
        // ).count();

        // double size2original_ms=std::chrono::duration<double,std::milli>(
        //     t8-t7
        // ).count();

        // double total_ms=std::chrono::duration<double,std::milli>(
        //     t8-t4
        // ).count();

        // double infer_ms=std::chrono::duration<double,std::milli>(
        //     t10-t9
        // ).count();

        // double cpu_wait_ms=std::chrono::duration<double,std::milli>(
        //     t12-t11
        // ).count();


        
        // postprocess_times.push_back(process_ms);
        // nms_times.push_back(nms_ms);         
        // size2original_times.push_back(size2original_ms);  
        // all_times.push_back(total_ms); 
        // infer_times.push_back(infer_ms);
        // output_ready_cpuwait_times.push_back(cpu_wait_ms);  
    }
    cudaStreamSynchronize(preprocess_stream);
    cudaStreamSynchronize(trtengine.stream);
}

void process_frames_signal(
    int RUNS,
    trtengine& trtengine,
    postprocess& pos,
    int dst_width,
    int dst_height
    )
{
     
    for(int frame_id=0;frame_id<RUNS;frame_id++)
    {
        FramSlot& slot=slots_[frame_id%2];
        slot.frame_id=frame_id;

        bool ok=cudaPreprocessV1(
            slot.device_src,
            slot.image.cols,
            slot.image.rows,
            slot.image.step,

            slot.device_input,
            dst_width,
            dst_height,
            
            slot.meta,
            trtengine.stream
        );
                    
        if(!ok)
        {
                std::cerr<<"fail to CUDA preprocess\n";
                return;
        }

        if(!trtengine.inferfram
            (
                slot.device_input,
                slot.device_output
            ))
            {
                std::cout<<"inferDevice"<<std::endl;
                return;
            }
//把结果转移到主机

        cudaMemcpyAsync(
            slot.host_output,
            slot.device_output,
            trtengine.output_bytes_,
            cudaMemcpyDeviceToHost,
            trtengine.stream
        );

        cudaEventRecord(slot.output_ready,trtengine.stream);

        cudaEventSynchronize(slot.output_ready);
//process
        std::vector<detection> det=pos.process(slot.host_output,slot.meta,confidence_threshold);

        const std::vector<detection> det2=det;
        
        std::vector<detection> det3=pos.nms(det2);

        for(int i=0;i<det3.size();i++)
            { 
                pos.size2original(det3[i],slot.meta.scale,slot.meta.pad_x,slot.meta.pad_y,slot.meta.original_width,slot.meta.original_height);
            }
    }
    cudaStreamSynchronize(trtengine.stream);
}

int main()
{
    RuntimeTiming timing;
    const int dst_width=640;
    const int dst_height=640;

    slots_[0].image=cv::imread(image1_PATH);

    if(slots_[0].image.empty())
    {
        std::cerr<<"fail to load image"<<std::endl;
        return -1;
    }

    if(!slots_[0].image.isContinuous())
    {
        slots_[0].image=slots_[0].image.clone();
    }

    slots_[1].image=cv::imread(image2_PATH);
    if(slots_[1].image.empty())
    {
        std::cerr<<"fail to load image"<<std::endl;
        return -1;
    }
    if(!slots_[1].image.isContinuous())
    {
        slots_[1].image=slots_[1].image.clone();
    }
//cpu preprocess,为了得到meta
    slots_[0].meta.new_size=640;
    if(!slots_[0].meta.computemeta(slots_[0].image))
    {
        std::cout<<"meta计算失误";
    }

    slots_[1].meta.new_size=640;
    if(!slots_[1].meta.computemeta(slots_[1].image))
    {
        std::cout<<"meta计算失误";
    }

// allocate GPU source
    //slot0

    

    trtengine trtengine;

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

    cudaMalloc(
        &slots_[0].device_output,
        trtengine.output_bytes_
    );

    cudaMalloc(
        &slots_[1].device_output,
        trtengine.output_bytes_
    );

    cudaStreamCreate(&trtengine.stream);

    cudaMallocHost(
        reinterpret_cast<void**>(&slots_[0].host_output),
        trtengine.output_bytes_
    );

    cudaMallocHost(
        reinterpret_cast<void**>(&slots_[1].host_output),
        trtengine.output_bytes_
    );

    postprocess pos{
        confidence_threshold,
        iou_threshold
    };

    std::vector<detection> det_A;
    std::vector<detection> det_B;

    int mismatch=0;
    int processed=0;

    cudaEventCreateWithFlags(&slots_[0].preprocess_done,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&slots_[0].output_ready,cudaEventDisableTiming);

    cudaEventCreateWithFlags(&slots_[1].preprocess_done,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&slots_[1].output_ready,cudaEventDisableTiming);

    cudaStream_t preprocess_stream=nullptr;
    cudaStreamCreate(&preprocess_stream);


    int warmup=30;
    int runs=1000;

    // process_frames(
    //     warmup,
    //     trtengine,
    //     pos,
    //     preprocess_stream,
    //     dst_width,
    //     dst_height
    // );
    process_frames_signal(
        warmup,
        trtengine,
        pos,
        dst_width,
        dst_height
    );

    std::vector<double>postprocess_times;      
    std::vector<double>nms_times;            
    std::vector<double>size2original_times;    
    std::vector<double>all_times;   
    std::vector<double>infer_times;
    std::vector<double>output_ready_cpuwait_times;

    postprocess_times.reserve(runs);
    nms_times.reserve(runs);
    size2original_times.reserve(runs);   
    all_times.reserve(runs);

    infer_times.reserve(runs); 
    output_ready_cpuwait_times.reserve(runs); 

    auto t0=std::chrono::steady_clock::now();
    process_frames(
        runs,
        trtengine,
        pos,
        preprocess_stream,
        dst_width,
        dst_height,

        postprocess_times,     
        nms_times,           
        size2original_times,
        all_times,

        infer_times,
        output_ready_cpuwait_times
    );

    // process_frames_signal(
    //     runs,
    //     trtengine,
    //     pos,
    //     dst_width,
    //     dst_height
    // );
    auto t1=std::chrono::steady_clock::now();

    double total_ms=
        std::chrono::duration<double,std::milli>(
            t1-t0
        ).count();
    
    double throughtput=1000.0/(total_ms/1000.0);

    std::cout<<"total:"<<total_ms<<"ms\n"<<std::endl;
    std::cout<<"throughput:"<<throughtput<<"FPS\n"<<std::endl;
    
    // std::cout<<"postprocess_times:"<<std::endl;
    // coutime(postprocess_times);
    // std::cout<<"nms_times:"<<std::endl;
    // coutime(nms_times);
    // std::cout<<"size2original_times:"<<std::endl;
    // coutime(size2original_times);
    // std::cout<<"all_times:"<<std::endl;
    // coutime(all_times);
    // std::cout<<"infer_times:"<<std::endl;
    // coutime(infer_times);
    // std::cout<<"output_ready_cpuwait_times:"<<std::endl;
    // coutime(output_ready_cpuwait_times);

    for(auto& slot : slots_)
    {
        cudaHostUnregister(slot.image.data);

        cudaFree(slot.device_input);
        cudaFree(slot.device_output);

        cudaFreeHost(slot.host_output);
        
        slot.device_input=nullptr;
        slot.device_output=nullptr;
        slot.host_output=nullptr;
        slot.host_output=nullptr;
    }

    cudaStreamDestroy(trtengine.stream);

    return 0;

//     FramSlot& slot0=slots_[0];

//     bool ok=cudaPreprocessV1(
//             slot0.device_src,
//             slot0.image.cols,
//             slot0.image.rows,
//             slot0.image.step,

//             slot0.device_input,
//             dst_width,
//             dst_height,
            
//             slot0.meta,
//             preprocess_stream
//         );
                
//     if(!ok)
//     {
//             std::cerr<<"fail to CUDA preprocess\n";
//             return -1;
//     }

//     cudaEventRecord(slot0.preprocess_done,preprocess_stream);

//     for(int frame_id=0;frame_id<1000;frame_id++)
//     {
//         processed++;
//         FramSlot& current=slots_[frame_id%2];
//         current.frame_id=frame_id;

//         cudaStreamWaitEvent(
//             trtengine.stream,
//             current.preprocess_done,
//             0
//         );

//         if(!trtengine.inferfram
//             (
//                 current.device_input,
//                 current.device_output
//             ))
//             {
//                 std::cout<<"inferDevice"<<std::endl;
//                 return -1;
//             }


// //把结果转移到主机

//         cudaMemcpyAsync(
//             current.host_output,
//             current.device_output,
//             trtengine.output_bytes_,
//             cudaMemcpyDeviceToHost,
//             trtengine.stream
//         );

//         if(frame_id+1<1000)
//         {
//             FramSlot& next=slots_[(frame_id+1)%2];
//             next.frame_id=frame_id+1;
//             bool ok=cudaPreprocessV1(
//                     next.device_src,
//                     next.image.cols,
//                     next.image.rows,
//                     next.image.step,

//                     next.device_input,
//                     dst_width,
//                     dst_height,
                    
//                     next.meta,
//                     preprocess_stream
//                 );
            
//             if(!ok)
//             {
//                     std::cerr<<"fail to CUDA preprocess\n";
//                     return -1;
//             }

//             cudaEventRecord(
//                 next.preprocess_done,
//                 preprocess_stream
//             );
//         }

//         cudaEventRecord(current.output_ready,trtengine.stream);

//         cudaEventSynchronize(current.output_ready);
// //process
//         std::vector<detection> det=pos.process(current.host_output,current.meta,confidence_threshold);

//         const std::vector<detection> det2=det;
        
//         std::vector<detection> det3=pos.nms(det2);

//         for(int i=0;i<det3.size();i++)
//             { 
//                 pos.size2original(det3[i],current.meta.scale,current.meta.pad_x,current.meta.pad_y,current.meta.original_width,current.meta.original_height);
//             }
//         if(frame_id==0)
//         {
//             det_A=det3;
//         }
//         else if(frame_id==1)
//         {
//             det_B=det3;
//         }
//         else if(frame_id%2==0)
//         {
//             if(det3!=det_A)
//             { 
//                 mismatch++;
//                 std::cout<<"A mistake at frame";
//             }
            
//         }
//         else
//         {
//             if(det3!=det_B)
//             { 
//                 mismatch++;
//                 std::cout<<"B mistake at frame";
//             }
//         }
// //画图
//         if(frame_id==239||frame_id==876)
//         {        
//             cv::Mat vis=current.image.clone(); 
//             for(int i=0;i<det3.size();i++)
//             { 
            
//                 cv::Point topLeft(det3[i].x1, det3[i].y1);
//                 cv::Point bottomRight(det3[i].x2, det3[i].y2);
//                 cv::rectangle(vis, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2); // 蓝色，线宽2[reference:8]
//                 std::string label;
//                 std::ostringstream oss;
//                 oss << std::fixed << std::setprecision(2) << det3[i].confidence;
//                 if(det3[i].class_id==0)
//                     label = "person "+oss.str();
//                 else
//                     label = "umbrella "+oss.str();
//                 int fontFace = cv::FONT_HERSHEY_COMPLEX;
//                 double fontScale = 1.5;
//                 cv::Scalar color(0, 0, 255); // 红色
//                 int thickness = 2;

//                 // 计算文字位置：左上角点 + 偏移（这里选择在框内偏移 5, 20 让文字显示在矩形内左上角）
//                 cv::Point textOrg(topLeft.x, topLeft.y -5);
//                 cv::putText(vis, label, textOrg, fontFace, fontScale, color, thickness);
//             }
//             std::string path = "results/frame_slots/test_" + std::to_string(frame_id)+".png";

//             cv::imwrite(path, vis);
//         }
//     }
//     cudaStreamSynchronize(trtengine.stream);


//     std::cout<<mismatch<<std::endl;
//     std::cout<<processed<<std::endl;
//     bool flag=det_A==det_B;
//     std::cout<<flag<<std::endl;

}