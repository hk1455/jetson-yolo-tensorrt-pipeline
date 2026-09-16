
#include "../include/preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include<cstring>
#include<algorithm>
#include<cmath>
#include<cstddef>
#include<iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>


void saveLetterboxMeta(const letterboxmeta& meta, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Failed to open meta file: " + path);
    }
    ofs << std::fixed << std::setprecision(8);
    ofs << "{\n";
    ofs << "  \"original_width\": " << meta.original_width << ",\n";
    ofs << "  \"original_height\": " << meta.original_height << ",\n";
    ofs << "  \"input_width\": " << meta.resized_width << ",\n";
    ofs << "  \"input_height\": " << meta.resized_height << ",\n";
    ofs << "  \"scale\": " << meta.scale << ",\n";
    ofs << "  \"pad_x\": " << meta.pad_x << ",\n";
    ofs << "  \"pad_y\": " << meta.pad_y << "\n";
    ofs << "}\n";
};

        
bool letterboxmeta::computemeta(const cv::Mat& image)
{
    if(image.empty())
        return false;

    int width=image.cols;
    int height=image.rows;

    original_height=height;
    original_width=width;

    scale=std::min(
      static_cast<float>(new_size)/height,
      static_cast<float>(new_size)/width
    );
    
    resized_height=std::round(scale*height);
    resized_width=std::round(scale*width);

    int left=(new_size-resized_width)/2;
    int top=(new_size-resized_height)/2;

    int right=new_size-resized_width-left;
    int bottom=new_size-resized_height-top;

    pad_x=left;
    pad_y=top;

    return true;
}

bool preprocess(const cv::Mat& image,std::vector<float>& tensor,letterboxmeta& meta)
{
    if(image.empty())
        return false;

    int width=image.cols;
    int height=image.rows;

    meta.original_height=height;
    meta.original_width=width;

    meta.scale=std::min(
      static_cast<float>(meta.new_size)/height,
      static_cast<float>(meta.new_size)/width
    );
    
    meta.resized_height=std::round(meta.scale*height);
    meta.resized_width=std::round(meta.scale*width);

    int left=(meta.new_size-meta.resized_width)/2;
    int top=(meta.new_size-meta.resized_height)/2;

    int right=meta.new_size-meta.resized_width-left;
    int bottom=meta.new_size-meta.resized_height-top;

    meta.pad_x=left;
    meta.pad_y=top;

    cv::Mat resized_image;
    cv::Mat paded_image;
    cv::Mat rgb; 
    cv::Mat rgb_float;

    cv::resize(
        image,
        resized_image,
        cv::Size(meta.resized_width,meta.resized_height),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    cv::copyMakeBorder(
        resized_image,
        paded_image,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114,114,114)
    );

  cv::cvtColor(
    paded_image,
    rgb,
    cv::COLOR_BGR2RGB
  );

  rgb.convertTo(
    rgb_float,
    CV_32FC3,
    1.0/255.0
  );


    int channel=3;
    int channel_size=meta.new_size*meta.new_size;
    tensor.resize(channel*channel_size);
    std::vector<cv::Mat> channels_mats;

    cv::split(rgb_float,channels_mats);

    for(int c=0;c<channel;c++){
        memcpy(
            tensor.data()+c*channel_size,
            channels_mats[c].data,
            channel_size*sizeof(float)
        );
    }
    
    return true;
};