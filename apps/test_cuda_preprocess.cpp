#include <vector>
#include<fstream>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "preprocess.hpp"
#include "cuda_preprocess.hpp"


#define OUT_PATH "results/cuda_preprocess/home_input_V6.bin"

double mean(const std::vector<double>& values){
    double sum=0.0;

    for(double v:values)
    {
        sum+=v;
    }
    return sum/values.size();
};

double findp(std::vector<double> data,double p)
{
    if(data.empty())
        return 0.0;
    size_t n=data.size();
    double rank=p*(n-1);
    size_t lo = static_cast<size_t>(std::floor(rank));
    size_t hi = static_cast<size_t>(std::ceil(rank));

    std::nth_element(data.begin(),data.begin()+lo,data.end());
    double lon=data[lo];

    if(lo==hi)
        return lon;

    std::nth_element(data.begin(),data.begin()+hi,data.end());
    double hin=data[hi];

    return hin-(hin-lon)*(hi-rank);
   
};

double findmin(std::vector<double> data)
{
    if(data.empty())
        return 0.0;

    std::nth_element(data.begin(),data.begin(),data.end());
    double min=data[0];

    return min;
};

double findmax(std::vector<double> data)
{
    if(data.empty())
        return 0.0;

    std::nth_element(data.begin(),data.end()-1,data.end());
    double max=data[4999];

    return max;
};

int main()
{
    const int dst_width=640;
    const int dst_height=640;

    constexpr int WARMUP=5000;
    constexpr int RUNS=5000;
    std::vector<double> ms;
    ms.reserve(RUNS);

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
    cudaEvent_t start,end;

    cudaEventCreate(&start);

    cudaEventCreate(&end);

//cpu preprocess,为了得到meta
    letterboxmeta meta;
    meta.new_size=640;
    std::vector<float> cpu_output;

    preprocess(
        image,
        cpu_output,
        meta
    );

// allocate GPU source

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
//2.texture memory
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

//3.texture memory(控制变量版)
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
    cudaMallocPitch(
        reinterpret_cast<void**>(&device_src_v3),
        &device_pitch_v3,
        width_bytes_v3,
        image_bgra.rows
    );
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

    cudaCreateTextureObject(
        &tex_obj_v3,
        &res_desc_v3,
        &tex_desc_v3,
        nullptr
    );


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


    //2.runs
    for(int i=0;i<RUNS;i++)
    { 
        cudaEventRecord(
            start,
            stream
        );

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
        cudaEventRecord(
            end,
            stream
        );

        cudaEventSynchronize(end);

        float ms1=0.0;
        
        cudaEventElapsedTime(
            &ms1,
            start,
            end
        );

        ms.push_back(ms1);
        if(!ok)
        {
            std::cerr<<"fail to CUDA preprocess\n";
            return -1;
        }
    }
//copy result back

    std::vector<float> cuda_output(output_element);

    cudaMemcpyAsync(
        cuda_output.data(),
        device_dst,
        output_bytes,
        cudaMemcpyDeviceToHost,
        stream
    );

    cudaStreamSynchronize(stream);

    std::cout
        <<"CUDA output elements:"
        <<cuda_output.size()
        <<"\n";

    std::cout
        <<"mean:  "
        << "cuda preprocessV4_<32,8>: " <<mean(ms) << " ms\n";

    std::cout
        <<"p50:  "
        << "cuda preprocessV4_<32,8>: " <<findp(ms,0.5) << " ms\n";

    std::cout
        <<"p95:  "
        << "cuda preprocessV4_<32,8>: " <<findp(ms,0.95) << " ms\n";
    std::cout
        <<"min:  "
        << "cuda preprocessV4_<32,8>: " <<findmin(ms) << " ms\n";
    std::cout
        <<"max:  "
        << "cuda preprocessV4_<32,8>: " <<findmax(ms) << " ms\n";


//把结果保存成bin文件
    std::ofstream outfile(OUT_PATH,std::ios::binary);
    if(!outfile)
        std::cout<<"打开输出路径错误"<<std::endl;    

    outfile.write(
        reinterpret_cast<char*>(cuda_output.data()),
        cuda_output.size()*sizeof(float)
    );



//release
    cudaFree(device_dst);
    cudaFree(device_src);
    cudaFree(device_src_v2);
     cudaFree(device_src_v3);
    
    cudaDestroyTextureObject(tex_obj_v2);
    cudaDestroyTextureObject(tex_obj_v3);
    cudaStreamDestroy(stream);

    return 0;

}