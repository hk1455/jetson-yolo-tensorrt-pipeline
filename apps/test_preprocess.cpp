#include "../include/preprocess.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define IMAGE_PATH "assets/home.jpeg"

#define OUT_PATH "results/cpp/home_input.bin"
#define JSON_OUT_PATH "results/cpp/home_input.json"

int main(){
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

    cv::Mat image;
    image=cv::imread(IMAGE_PATH,cv::IMREAD_COLOR);

    std::vector<float> tensor;

    bool suc=preprocess(image,tensor,meta);

    if(!suc)
        std::cout<<"前处理错误"<<std::endl;
    
    std::ofstream outfile(OUT_PATH,std::ios::binary);
    if(!outfile)
        std::cout<<"打开输出路径错误"<<std::endl;    

    outfile.write(
        reinterpret_cast<char*>(tensor.data()),
        tensor.size()*sizeof(float)
    );
    saveLetterboxMeta(meta,JSON_OUT_PATH);

    outfile.close();
}