#include <iostream>
#include <vector>
#include<fstream>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include "engine_builder.hpp"
#include "trt_engine.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"

#define MODEL_PATH "models/yolov8n_cpp_fp32.engine"
#define IMAGE_PATH "assets/home.jpeg"
#define OUT_PUT "results/cpp/home_detection.jpeg"

float confidence_threshold=0.20;
float iou_threshold=0.70;


int main(){
    RuntimeTiming timing;
    std::vector<float> input;
    std::vector<float> output;
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


    bool suc=preprocess(image1,input,meta);

    if(!suc)
    {
        std::cout<<"前处理错误"<<std::endl;
        return -1;
    }


//runengine
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
    output.resize(trtengine.output_elements_);
    if(!trtengine.infer(input.data(),output.data(),&timing))
    {
        std::cout<<"推理出错"<<std::endl;
        return -1;
    }

//postprocess
    postprocess pos{
        confidence_threshold,
        iou_threshold
    };

    std::cout<<output.size()<<std::endl;
    std::vector<detection> det=pos.process(output.data(),meta,confidence_threshold);

    const std::vector<detection> det2=det;
    std::vector<detection> det3=pos.nms(det2);

    for(int i=0;i<det3.size();i++)
    { 
        pos.size2original(det3[i],meta.scale,meta.pad_x,meta.pad_y,meta.original_width,meta.original_height);
    }

//画图

    for(int i=0;i<det3.size();i++)
    { 
        std::cout<<det3[i].class_id<<std::endl;
    }

    for(int i=0;i<det3.size();i++)
    { 
    
        cv::Point topLeft(det3[i].x1, det3[i].y1);
        cv::Point bottomRight(det3[i].x2, det3[i].y2);
        cv::rectangle(image1, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2); // 蓝色，线宽2[reference:8]
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
        cv::putText(image1, label, textOrg, fontFace, fontScale, color, thickness);
    }

    cv::imwrite(OUT_PUT, image1);
    return 0;

}
