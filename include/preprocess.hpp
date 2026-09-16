#pragma once

#include <opencv2/core.hpp>

#include <vector>


class letterboxmeta{
    public:
        int original_width=0;
        int original_height=0;
        int resized_width=0;
        int resized_height=0;

        int new_size=0;
 
        int pad_x=0;
        int pad_y=0;

        float scale=0.0f;
        
        bool computemeta(const cv::Mat& image);
};


bool preprocess(
    const cv::Mat& image,
    std::vector<float>& tensor,
    letterboxmeta& meta
);

void saveLetterboxMeta(const letterboxmeta& meta, const std::string& path);