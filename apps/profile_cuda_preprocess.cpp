#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "preprocess.hpp"
#include "cuda_preprocess.hpp"

int main()
{
    const int dst_width=640;
    const int dst_height=640;

    constexpr int WARMUP=5;
    constexpr int RUNS=5;

    cv::Mat image=cv::imread("assets/home.jpeg");

    if(image.empty())
    {
        std::cerr<<"fail to load image"<<std::endl;
        return -1;
    }

    if(!image.isContinuous())
    {
        image=image.clone();
    }

//compute meta
    letterboxmeta meta;
    meta.new_size=640;
    std::vector<float> cpu_output;

    if(!meta.computemeta(image))
    {
        std::cout<<"meta计算失误";
    }



// allocate GPU source
    //1.normal 
    uint8_t* device_src=nullptr;
    size_t src_bytes=
        image.step*image.rows;
    
    cudaMalloc(
        reinterpret_cast<void**>(&device_src),
        src_bytes
    );

    cudaMemcpy(
        device_src,
        image.data,
        src_bytes,
        cudaMemcpyHostToDevice
    );

//2.texture memory(控制变量版)
    uint8_t* device_src_v2=nullptr;
    size_t device_pitch=0;

    size_t width_bytes =
        static_cast<size_t>(image.cols)*3;
    //在gpu上分配位置
    cudaMallocPitch(
        reinterpret_cast<void**>(&device_src_v2),
        &device_pitch,
        width_bytes,
        image.rows
    );
    //把图片从cpu中搬到gpu里
    cudaMemcpy2D(
        device_src_v2,
        device_pitch,

        image.data,
        image.step,

        width_bytes,
        image.rows,

        cudaMemcpyHostToDevice
    );
    //保存texture memory的相关信息
    cudaResourceDesc res_desc_v2{};

    res_desc_v2.resType=
        cudaResourceTypePitch2D;

    res_desc_v2.res.pitch2D.devPtr=
        device_src_v2;

    res_desc_v2.res.pitch2D.desc=
        cudaCreateChannelDesc<unsigned char>();

    res_desc_v2.res.pitch2D.width=
        width_bytes;

    res_desc_v2.res.pitch2D.height=
        image.rows;

    res_desc_v2.res.pitch2D.pitchInBytes=
        device_pitch;

    //保存texture memory的使用边界
    cudaTextureDesc tex_desc_v2{};
    
    tex_desc_v2.addressMode[0]=
        cudaAddressModeClamp;//0代表x方向，clamp代表越界时卡在边缘

    tex_desc_v2.addressMode[1]=
        cudaAddressModeClamp;
    
    tex_desc_v2.filterMode=
        cudaFilterModePoint;//只取一个textel，不让texture帮我做bilinear
    
    tex_desc_v2.readMode=
        cudaReadModeElementType;//内存里是什么类型，就按什么类型返回

    tex_desc_v2.normalizedCoords=0;//使用真实texel坐标，而不是0-1的归一化坐标

    //创建真正的texture object
    cudaTextureObject_t tex_obj_v2=0;

    cudaCreateTextureObject(
        &tex_obj_v2,
        &res_desc_v2,
        &tex_desc_v2,
        nullptr
    );

//3.texture memory
    //使用texture memory的bilinear
    cv::Mat image_bgra;

    cv::cvtColor(
        image,
        image_bgra,
        cv::COLOR_BGR2BGRA
    );

    uchar4* device_src_v3=nullptr;
    size_t device_pitch_v3=0;

    size_t width_bytes_v3=
        static_cast<size_t>(image_bgra.cols)*sizeof(uchar4);
    //在gpu上分配位置
    cudaError_t err=cudaMallocPitch(
        reinterpret_cast<void**>(&device_src_v3),
        &device_pitch_v3,
        width_bytes_v3,
        image_bgra.rows
    );
    if(err!=cudaSuccess){
        std::cerr
            <<"cudaMallocPitch fail: "
            <<cudaGetErrorString(err)
            <<std::endl;
        
        return false;
    }
    
    //把图片从cpu中搬到gpu里
    cudaMemcpy2D(
        device_src_v3,
        device_pitch_v3,

        image_bgra.data,
        image_bgra.step,

        width_bytes_v3,
        image_bgra.rows,

        cudaMemcpyHostToDevice
    );
    //保存texture memory的相关信息
    cudaResourceDesc res_desc_v3{};

    res_desc_v3.resType=
        cudaResourceTypePitch2D;

    res_desc_v3.res.pitch2D.devPtr=
        device_src_v3;

    res_desc_v3.res.pitch2D.desc=
        cudaCreateChannelDesc<uchar4>();

    res_desc_v3.res.pitch2D.width=
        image_bgra.cols;

    res_desc_v3.res.pitch2D.height=
        image_bgra.rows;

    res_desc_v3.res.pitch2D.pitchInBytes=
        device_pitch_v3;

    //保存texture memory的使用边界
    cudaTextureDesc tex_desc_v3{};
    
    tex_desc_v3.addressMode[0]=
        cudaAddressModeClamp;//0代表x方向，clamp代表越界时卡在边缘

    tex_desc_v3.addressMode[1]=
        cudaAddressModeClamp;
    
    tex_desc_v3.filterMode=
        cudaFilterModeLinear;//让texture做bilinear
    
    tex_desc_v3.readMode=
        cudaReadModeNormalizedFloat;//返回float

    tex_desc_v3.normalizedCoords=0;//使用真实texel坐标，而不是0-1的归一化坐标

    //创建真正的texture object
    cudaTextureObject_t tex_obj_v3=0;

    cudaError_t err1=cudaCreateTextureObject(
        &tex_obj_v3,
        &res_desc_v3,
        &tex_desc_v3,
        nullptr
    );
    if(err1!=cudaSuccess){
        std::cerr
            <<"cudaCreateTextureObject fail: "
            <<cudaGetErrorString(err1)
            <<std::endl;
            
        return false;
    }


//allcote GPU output
    size_t output_element=3ULL*dst_height*dst_width;

    size_t output_bytes=output_element*sizeof(float);

    float* device_dst=nullptr;

    cudaMalloc(
        reinterpret_cast<void**>(&device_dst),
        output_bytes
    );

//create stream
    
    cudaStream_t stream;

    cudaStreamCreate(&stream);

//lauch CUDA preprocess
    //1.warmup
    for(int i=0;i<WARMUP;i++)
    { 
        bool ok=cudaPreprocessV6(
            device_src,
            image.cols,
            image.rows,
            image.step,

            device_dst,
            dst_width,
            dst_height,
            
            meta,
            stream
        );

        if(!ok)
        {
            std::cerr<<"fail to CUDA preprocess\n";
            return -1;
        }
    }
    cudaStreamSynchronize(stream);

    //2.runs
    for(int i=0;i<RUNS;i++)
    { 

        bool ok=cudaPreprocessV6(
            device_src,
            image.cols,
            image.rows,
            image.step,

            device_dst,
            dst_width,
            dst_height,
            
            meta,
            stream
        );

        if(!ok)
        {
            std::cerr<<"fail to CUDA preprocess\n";
            return -1;
        }
    }
    cudaError_t err2;
    err2=cudaStreamSynchronize(stream);
    if(err2!=cudaSuccess){
        std::cerr
            <<"kernel execution error: "
            <<cudaGetErrorString(err2)
            <<std::endl;
            
        return false;
    }

    //release
    cudaFree(device_dst);
    cudaFree(device_src);
    cudaFree(device_src_v2);
    cudaFree(device_src_v3);
    cudaDestroyTextureObject(tex_obj_v3);
    cudaDestroyTextureObject(tex_obj_v2);

    cudaStreamDestroy(stream);

    return 0;
}