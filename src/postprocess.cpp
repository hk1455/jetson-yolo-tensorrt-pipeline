#include "postprocess.hpp"
#include <iostream>
#include<algorithm>
#include<cmath>
#include<numeric>
#include<stdexcept>
#include<fstream>
#include<string>

bool postprocess::readbin(const std::string path,std::vector<float>& raw_output)
{
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "文件打开失败！" << std::endl;
        return false;
    }

    int output_size=file.tellg();
    file.seekg(0,std::ios::beg);
    raw_output.resize(output_size/sizeof(float));
    file.read (reinterpret_cast<char*>(raw_output.data()),output_size);
    return true;
};

void postprocess::xywh_xyxy(
        float x,
        float y,
        float w,
        float h,
        detection& det
){
    det.x1=x-w/2;
    det.y1=y-h/2;
    det.x2=x+w/2;
    det.y2=y+h/2;

};

float postprocess::compute_iou(
    const detection& a,
    const detection& b
){
    float max_x1=std::max(a.x1,b.x1);
    float max_y1=std::max(a.y1,b.y1);

    float min_x2=std::min(a.x2,b.x2);
    float min_y2=std::min(a.y2,b.y2);

    float z=0;

    float over_x=std::max(min_x2-max_x1,z);
    float over_y=std::max(min_y2-max_y1,z);

    float over_s=over_x*over_y;
    float x1=(a.x2-a.x1)*(a.y2-a.y1);
    float x2=(b.x2-b.x1)*(b.y2-b.y1);

    return over_s/(x1+x2-over_s);
    
}

std::vector<detection> postprocess::nms(
    const std::vector<detection>& det
)
{

    std::vector<detection> det1;
    det1=det;
    std::sort(det1.begin(),det1.end(),
            [](const detection& a,const detection& b)
            {
                return a.confidence>b.confidence;
            }
        );

    size_t i=0;
    while (i<det1.size())
    {
        size_t j=i+1;
        while(j<det1.size())
        {
            if(det1[j].class_id==det1[i].class_id)
            {
                float iou=compute_iou(det1[j],det1[i]);
                    if(iou>iou_threshold)
                        det1.erase(det1.begin()+j);
                    else
                        j++;
            }
            else 
                j++;
        }
        i++;
    }

    return det1;
    
};

std::vector<detection> postprocess::process(float* output,letterboxmeta meta,float confidence_threshold)
{

    std::vector<detection> det;
    for(int i=0;i<8400;i++){
        float max_confidence=output[i+4*8400];
        int class_id=0;
        
        for(int j=0;j<80;j++)
        {
            float new_confidence=output[(4+j)*8400+i];
            if(max_confidence<new_confidence)
            {
                max_confidence=new_confidence;
                class_id=j;
            }
        }

        if(max_confidence>confidence_threshold)
        {
            detection det1;
            xywh_xyxy(
                output[i],
                output[i+8400],
                output[i+2*8400],
                output[i+3*8400],
                det1
            );
            det1.confidence=max_confidence;
            det1.class_id=class_id;
            det.push_back(det1);
        }
    }
    return det;
};

void postprocess::size2original(
        detection& det,
        float scale,
        int pad_x,
        int pad_y,
        int original_width,
        int original_height
){
    det.x1=(det.x1-pad_x)/scale;
    det.x2=(det.x2-pad_x)/scale;
    det.y1=(det.y1-pad_y)/scale;
    det.y2=(det.y2-pad_y)/scale;

    float z=0;
    det.x1=std::max(z,det.x1);
    det.x2=std::max(z,det.x2);
    det.y1=std::max(z,det.y1);
    det.y2=std::max(z,det.y2);
};