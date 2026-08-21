#pragma once

#include<vector>
#include<string>

struct detection{
    public:
        float x1=0.0;
        float y1=0.0;
        float x2=0.0;
        float y2=0.0;

    float confidence=0.0;
    int class_id=0;
};


class postprocess{
    public:
        float conf_threshold=0.0;
        float iou_threshold=0.0;
    
    bool readbin(const std::string path,std::vector<float>& output);

    void xywh_xyxy(
        float x,
        float y,
        float w,
        float h,
        detection& det
    );

    float compute_iou(
        const detection& a,
        const detection& b
    );

    std::vector<detection>  nms(
        const std::vector<detection>& detection
    );

    void size2original(
        detection& det,
        float scale,
        int pad_x,
        int pad_y,
        int original_width,
        int original_height
    );
};