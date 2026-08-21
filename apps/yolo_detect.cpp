#include <iostream>
#include <vector>
#include<fstream>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include "engine_builder.hpp"
#include "trt_engine.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"

#define IMAGE_PATH "assets/home.jpeg"
#define OUT_PUT "results/cpp/home_detection.jpeg"

#define BINOUT_PATH "results/cpp/home_input.bin"

const char* In_BIN_PATH="results/cpp/home_input.bin";
const char* out_BIN_PATH="results/cpp/home_raw_output.bin";


#define output_path "results/cpp/home_raw_output.bin"

float confidence_threshold=0.20;
float iou_threshold=0.70;


int main(){
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

    std::vector<float> tensor;

    bool suc=preprocess(image1,tensor,meta);

    if(!suc)
        std::cout<<"前处理错误"<<std::endl;
    
    std::ofstream outfile(BINOUT_PATH,std::ios::binary);
    if(!outfile)
        std::cout<<"打开输出路径错误"<<std::endl;    

    outfile.write(
        reinterpret_cast<char*>(tensor.data()),
        tensor.size()*sizeof(float)
    );

    outfile.close();

    std::string MODEL_PATH="models/yolov8n_cpp_fp32.engine";

//runengine
    trtengine trtengine;
    trtengine.load(MODEL_PATH);
    trtengine.printTensorInfo();

    trtengine.createContext();
    size_t in_size;
    int batch=1;
    trtengine.readFloatBinary(In_BIN_PATH,in_size);
    trtengine.infer(batch,out_BIN_PATH);


    postprocess pos{
        confidence_threshold,
        iou_threshold
    };
    std::vector<float> raw_output;

    pos.readbin(output_path,raw_output);

    std::vector<detection> det;
//筛掉confidence小于最小值的检测框  
    for(int i=0;i<8400;i++){
        float max_confidence=raw_output[i+4*8400];
        int class_id=0;
        for(int j=0;j<80;j++)
            {
                float new_confidence=raw_output[(4+j)*8400+i];
                if(max_confidence<new_confidence)
                {
                    max_confidence=new_confidence;
                    class_id=j;
                }
            }
        if(max_confidence>confidence_threshold)
        {
            detection det1;
            pos.xywh_xyxy(
                raw_output[i],
                raw_output[i+8400],
                raw_output[i+2*8400],
                raw_output[i+3*8400],
                det1
            );
            det1.confidence=max_confidence;
            det1.class_id=class_id;
            det.push_back(det1);
        }
    }

    const std::vector<detection> det2=det;
    std::vector<detection> det3=pos.nms(det2);

    for(int i=0;i<det3.size();i++)
    { 
        pos.size2original(det3[i],meta.scale,meta.pad_x,meta.pad_y,meta.original_width,meta.original_height);
    }

//画图
    cv::Mat image=cv::imread(IMAGE_PATH);
    if(image.empty())
        {
            std::cerr<<"图片加载失败"<<std::endl;
            return -1;
        }

    for(int i=0;i<det3.size();i++)
    { 
        std::cout<<det3[i].class_id<<std::endl;
    }

    for(int i=0;i<det3.size();i++)
    { 
    
        cv::Point topLeft(det3[i].x1, det3[i].y1);
        cv::Point bottomRight(det3[i].x2, det3[i].y2);
        cv::rectangle(image, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2); // 蓝色，线宽2[reference:8]
        std::string label;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << det3[i].confidence;
        if(det3[i].class_id==0)
            label = "person "+oss.str();
        else
            label = "umbrella "+oss.str();
        int fontFace = cv::FONT_HERSHEY_COMPLEX;
        double fontScale = 1.5;
        cv::Scalar color(0, 0, 255); // 红色
        int thickness = 2;

        // 计算文字位置：左上角点 + 偏移（这里选择在框内偏移 5, 20 让文字显示在矩形内左上角）
        cv::Point textOrg(topLeft.x, topLeft.y -5);
        cv::putText(image, label, textOrg, fontFace, fontScale, color, thickness);
    }

    cv::imwrite(OUT_PUT, image);
    return 0;

}