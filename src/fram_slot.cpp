#include<cstdint>
#include<cstddef>
#include <condition_variable>
#include<algorithm>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <nvtx3/nvToolsExt.h>
#include <NvInfer.h>
#include <deque>

#include "fram_slot.hpp"
#include "cuda_postprocess.hpp"


void captureLoop(
    const std::string& video_path,
    BoundedQueue<int>& free_slot,
    BoundedQueue<FramePacket>& frame_queue,
    std::array<FramSlot, 2>& slots
)
{
 
    cv::VideoCapture cap(video_path);

    if(!cap.isOpened())
    {
        std::cerr<<"fail to open vedio\n";

        frame_queue.close();
        return;
    }

    int64_t frame_id=0;

    while(true)
    {
        int slot_index=-1;

        {     
            NvtxRange range("Queue Wait/Free Slot");
            if(!free_slot.pop(slot_index))
            {
                break;
            }
        }

        FramSlot& slot=slots[slot_index];

        {        
            NvtxRange range("capture Grab");
            if(!cap.grab())
            {
                free_slot.pushBlocking(slot_index);

                break;
            }
        }

        {        
            NvtxRange range("capture Retrieve");
            if(!cap.retrieve(slot.image))
            {
                free_slot.pushBlocking(slot_index);

                break;
            }
        }


        slot.frame_id=frame_id++;

        slot.capture_time=Clock::now();


        FramePacket packet;
        packet.slot_index=slot_index;

        {
            NvtxRange range("Frame Queue Push");            
            if(!frame_queue.pushBlocking(std::move(packet)))
            {
        
                free_slot.pushBlocking(slot_index);
                break;
            }
        }

    }

    std::cout<<"Capture finished\n";
    frame_queue.close();
};


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
)

{
    FramePacket packet;

    while(true)
    {
        {
            NvtxRange range("Queue wait/Frame Pop");

            if(!frame_queue.pop(packet))
                break;
        }

        int slot_index=packet.slot_index;
        FramSlot& slot=slots[slot_index];

        bool ok=false;

       {
            NvtxRange range("submit Preprocess");

            ok=cudaPreprocessV1(
                    slot.device_src,
                    slot.image.cols,
                    slot.image.rows,
                    slot.image.step,

                    slot.device_input,
                    dst_width,
                    dst_height,
                    
                    slot.meta,
                    preprocess_stream
                );
        }
        
        if(!ok)
        {
                std::cerr<<"fail to CUDA preprocess\n";
                return;
        }

        {        
            NvtxRange range("CUDA Stream Dependency");

            cudaEventRecord(
                slot.preprocess_done,
                preprocess_stream
            );
            cudaStreamWaitEvent(
                trtengine.stream,
                slot.preprocess_done,
                0
            );
        }
        {
            NvtxRange range("Submit TensorRT");
            if(!trtengine.inferfram(slot.device_input, slot.device_output))
            {
                std::cerr<<"TensorRT submission failed\n";
                std::exit(EXIT_FAILURE);
            }
        }
        {
            NvtxRange range("D2H Raw Output");
            cuda_check(cudaMemcpyAsync(slot.host_output, slot.device_output,
                                       trtengine.output_bytes_, cudaMemcpyDeviceToHost,
                                       trtengine.stream));
        }
        cudaEventRecord(slot.output_ready,trtengine.stream);


        slot.gpu_submit_time=Clock::now();

        ResultPacket packet;
        packet.slot_index=slot_index;

        {
            NvtxRange range("Queue Wait/Result Push");
            if(!result_packet.pushBlocking(std::move(packet)))
            {
                cudaEventSynchronize(slot.output_ready);
                free_slot.pushBlocking(slot_index);

                break;
            }
        }

    }
    result_packet.close();
};



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
)
{
    ResultPacket packet;
    while(true)
   { 
        {
            NvtxRange range("Queue Wait/Result Pop");
            if(!result_packet.pop(packet))
            break;
        }

        int slot_index=packet.slot_index;
        FramSlot& slot=slots[slot_index];

        {
            NvtxRange range("Wait Output Ready");
            cudaEventSynchronize(slot.output_ready);
        }

        slot.post_start_time=Clock::now();

        //process
        {
            NvtxRange range("CPU Postprocess");
            const std::vector<detection> det2 =
                pos.process(slot.host_output, slot.meta, confidence_threshold);

            std::vector<detection> det3=pos.nms(det2);
            

            for(int i=0;i<det3.size();i++)
            { 
                pos.size2original(det3[i],slot.meta.scale,slot.meta.pad_x,slot.meta.pad_y,slot.meta.original_width,slot.meta.original_height);

                det3[i].x1=std::clamp(
                    det3[i].x1,
                    0.0f,
                    static_cast<float>(slot.meta.original_width)
                );

                det3[i].x2=std::clamp(
                    det3[i].x2,
                    0.0f,
                    static_cast<float>(slot.meta.original_width)
                );
            
                det3[i].y1=std::clamp(
                    det3[i].y1,
                    0.0f,
                    static_cast<float>(slot.meta.original_height)
                );

                det3[i].y2=std::clamp(
                    det3[i].y2,
                    0.0f,
                    static_cast<float>(slot.meta.original_height)
                );
            }
            processed_frames++;

            // for(auto& d :det3)
            // {
            //     std::cout<<d.confidence<<d.class_id<<std::endl;
            // }
            // cv::Mat vis=slot.image.clone(); 
            // for(int i=0;i<det3.size();i++)
            // { 
            
            //     cv::Point topLeft(det3[i].x1, det3[i].y1);
            //     cv::Point bottomRight(det3[i].x2, det3[i].y2);
            //     cv::rectangle(vis, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2); // 蓝色，线宽2[reference:8]
            //     std::string label;
            //     std::ostringstream oss;
            //     oss << std::fixed << std::setprecision(2) << det3[i].confidence;
            //     if(det3[i].class_id==0)
            //         label = "person "+oss.str();
            //     else
            //         label = "umbrella "+oss.str();
            //     int fontFace = cv::FONT_HERSHEY_COMPLEX;
            //     double fontScale = 1;
            //     cv::Scalar color(0, 0, 255); // 红色
            //     int thickness = 1;

            //     // 计算文字位置：左上角点 + 偏移（这里选择在框内偏移 5, 20 让文字显示在矩形内左上角）
            //     cv::Point textOrg(topLeft.x, topLeft.y -5);
            //     cv::putText(vis, label, textOrg, fontFace, fontScale, color, thickness);
            // }
            // writer.write(vis);
        free_slot.pushBlocking(packet.slot_index);
     }
        
    }
}
