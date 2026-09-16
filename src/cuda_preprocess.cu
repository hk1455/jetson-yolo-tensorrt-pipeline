#include "cuda_preprocess.hpp"
#include "fram_slot.hpp"
#include<iostream>
#include<cuda_runtime.h>
#include<cstdint>
//每个thread负责一个像素点
__global__ void preprocessKernalV0(
    const uint8_t*src,
    int src_width,
    int src_height,
    int src_stride,

    float* dst,
    int dst_width,
    int dst_height,

    int pad_x,
    int pad_y,
    int resize_x,
    int resize_y
)
{
//计算右下角的坐标
    int x=pad_x+resize_x;
    int y=pad_y+resize_y;
//dx和dy是当前thread在grid全局里的位置
    int dx=
        blockIdx.x*blockDim.x
        +threadIdx.x;
    int dy=
        blockIdx.y*blockDim.y
        +threadIdx.y;    
    if(dx>=dst_width||
        dy>=dst_height)
        {
            return;
        }

    int area=dst_height*dst_width;
    int idx=dy*dst_width+dx;

//判断是否属于pad区域
    if (dx<pad_x||dy>=y||dx>=x||dy<pad_y)
    {
        float gray_val = 114.0f / 255.0f; 
        dst[idx]=gray_val;
        dst[area+idx]=gray_val;
        dst[2*area+idx]=gray_val;
    }
    else
    {
//计算出原图中的位置，然后bgr转化成rgb
        float src_dx=(dx-pad_x+0.5f)*src_width/resize_x-0.5f;
        float src_dy=(dy-pad_y+0.5f)*src_height/resize_y-0.5f;

        int x0=floorf(src_dx);
        int y0=floorf(src_dy);

        int x1=x0+1;
        int y1=y0+1;

        float w00=(x1-src_dx)*(y1-src_dy);
        float w01=(src_dx-x0)*(y1-src_dy);
        float w10=(x1-src_dx)*(src_dy-y0);
        float w11=(src_dx-x0)*(src_dy-y0);
//clamp
        x0=max(0,x0);
        x1=min(x1,src_width-1);
        y0=max(0,y0);
        y1=min(y1,src_height-1);

        int index00=y0*src_stride+x0*3;
        int index01=y0*src_stride+x1*3;
        int index10=y1*src_stride+x0*3;
        int index11=y1*src_stride+x1*3;


        float r=src[index00+2]*w00+
                src[index01+2]*w01+
                src[index10+2]*w10+
                src[index11+2]*w11;

        float g=src[index00+1]*w00+
                src[index01+1]*w01+
                src[index10+1]*w10+
                src[index11+1]*w11;

        float b=src[index00]*w00+
                src[index01]*w01+
                src[index10]*w10+
                src[index11]*w11;

        int ri=__float2int_rn(r);
        int gi=__float2int_rn(g);
        int bi=__float2int_rn(b);

        ri=max(0,min(255,ri));
        gi=max(0,min(255,gi));
        bi=max(0,min(255,bi));
//直接截断
        // int r=src[index00+2]*w00+
        //         src[index01+2]*w01+
        //         src[index10+2]*w10+
        //         src[index11+2]*w11;

        // int g=src[index00+1]*w00+
        //         src[index01+1]*w01+
        //         src[index10+1]*w10+
        //         src[index11+1]*w11;

        // int b=src[index00]*w00+
        //         src[index01]*w01+
        //         src[index10]*w10+
        //         src[index11]*w11;


        dst[idx]=ri/ 255.0f;
        dst[area+idx]=gi/ 255.0f;
        dst[2*area+idx]=bi/ 255.0f;
    }
    
};

bool cudaPreprocessV0(    
    const uint8_t*device_src,
    int src_width,
    int src_height,
    int src_stride,

    float* device_dst,
    int dst_width,
    int dst_height,

    const letterboxmeta& meta,
    cudaStream_t stream
){
    dim3 block(16,16);

    dim3 grid(
        (dst_width+block.x-1)/block.x,
        (dst_height+block.y-1)/block.y
    );

    preprocessKernalV0<<<grid,block,0,stream>>>(
        device_src,
        src_width,
        src_height,
        src_stride,

        device_dst,
        dst_width,
        dst_height,

        meta.pad_x,
        meta.pad_y,
        meta.resized_width,
        meta.resized_height
    );

    cudaError_t err=
        cudaGetLastError();

    return err==cudaSuccess;

};


__global__ void preprocessKernalV1(
    const uint8_t* __restrict__ src,
    int src_width,
    int src_height,
    int src_stride,

    float* __restrict__ dst,
    int dst_width,
    int dst_height,

    int pad_x,
    int pad_y,
    int right,
    int bottom,
    int area,
    float pax_center,
    float x_scale,
    float pay_center,
    float y_scale,
    float stand,
    float gray_val
)
{

//dx和dy是当前thread在grid全局里的位置
    int dx=
        blockIdx.x*blockDim.x+threadIdx.x;
    int dy=
        blockIdx.y*blockDim.y+threadIdx.y;    
    if(dx>=dst_width||
        dy>=dst_height)
        {
            return;
        }

    int idx=dy*dst_width+dx;

//判断是否属于pad区域
    if (dx<pad_x||dy>=bottom||dx>=right||dy<pad_y)
    {
        dst[idx]=gray_val;
        dst[area+idx]=gray_val;
        dst[2*area+idx]=gray_val;
    }
    else
    {
//计算出原图中的位置，然后bgr转化成rgb
        float src_dx=(dx-pax_center)*x_scale-0.5f;
        float src_dy=(dy-pay_center)*y_scale-0.5f;
//四个顶点对应的坐标
        int x0=floorf(src_dx);
        int y0=floorf(src_dy);

        int x1=x0+1;
        int y1=y0+1;
//四个顶点对应的权重
        float w00=(x1-src_dx)*(y1-src_dy);
        float w01=(src_dx-x0)*(y1-src_dy);
        float w10=(x1-src_dx)*(src_dy-y0);
        float w11=(src_dx-x0)*(src_dy-y0);
//clamp
        x0=max(0,x0);
        x1=min(x1,src_width-1);
        y0=max(0,y0);
        y1=min(y1,src_height-1);
//四个顶点对应的存储位置
        int index00=y0*src_stride+x0*3;
        int index01=y0*src_stride+x1*3;
        int index10=y1*src_stride+x0*3;
        int index11=y1*src_stride+x1*3;


        float r=src[index00+2]*w00+
                src[index01+2]*w01+
                src[index10+2]*w10+
                src[index11+2]*w11;

        float g=src[index00+1]*w00+
                src[index01+1]*w01+
                src[index10+1]*w10+
                src[index11+1]*w11;

        float b=src[index00]*w00+
                src[index01]*w01+
                src[index10]*w10+
                src[index11]*w11;

        int ri=__float2int_rn(r);
        int gi=__float2int_rn(g);
        int bi=__float2int_rn(b);

        ri=max(0,min(255,ri));
        gi=max(0,min(255,gi));
        bi=max(0,min(255,bi));

        dst[idx]=ri*stand;
        dst[area+idx]=gi*stand;
        dst[2*area+idx]=bi*stand;
    }
    
};


bool cudaPreprocessV1(    
    const uint8_t*device_src,
    int src_width,
    int src_height,
    int src_stride,

    float* device_dst,
    int dst_width,
    int dst_height,

    const letterboxmeta& meta,
    cudaStream_t stream
){
    dim3 block(32,8);

    dim3 grid(
        (dst_width+block.x-1)/block.x,
        (dst_height+block.y-1)/block.y
    );

    int right=meta.pad_x+meta.resized_width;//可移出
    int bottom=meta.pad_y+meta.resized_height;//可移出

    int area=dst_height*dst_width;//可移出
    float pax_center=meta.pad_x-0.5f;
    float x_scale=src_width*1.0f/meta.resized_width;

    float pay_center=meta.pad_y-0.5f;
    float y_scale=src_height*1.0f/meta.resized_height;

    float stand=1.0f/255.0f;

    float gray_val = 114.0f / 255.0f;

   {
        NvtxRange range("cudapreprocess_kernel");
        preprocessKernalV1<<<grid,block,0,stream>>>(
            device_src,
            src_width,
            src_height,
            src_stride,

            device_dst,
            dst_width,
            dst_height,

            meta.pad_x,
            meta.pad_y,

            right,
            bottom,
            area,
            pax_center,
            x_scale,
            pay_center,
            y_scale,
            stand,
            gray_val
    );
}
    cudaError_t err=
        cudaGetLastError();

    if(err!=cudaSuccess)
    {
        std::cerr<<"preprocess launch error:"
        <<cudaGetErrorString(err)
        <<std::endl;

        return false;
    }

    return true;

};

//texture memory
__device__ __forceinline__ 
uint8_t loadBGR(
    cudaTextureObject_t tex,
    int x,
    int y,
    int c
)
{
    float tx=static_cast<float>(x*3+c)+0.5f;
    float ty=static_cast<float>(y)+0.5f;

    return tex2D<unsigned char>(
        tex,
        tx,
        ty
    );
}


//取值时使用texture，bilinear还是使用software
__global__ void preprocessKernalV2(
    cudaTextureObject_t src_tex,
    int src_width,
    int src_height,

    float* __restrict__ dst,
    int dst_width,
    int dst_height,

    int pad_x,
    int pad_y,
    int right,
    int bottom,
    int area,
    float pax_center,
    float x_scale,
    float pay_center,
    float y_scale,
    float stand,
    float gray_val
)
{

//dx和dy是当前thread在grid全局里的位置
    int dx=
        blockIdx.x*blockDim.x
        +threadIdx.x;
    int dy=
        blockIdx.y*blockDim.y
        +threadIdx.y;    
    if(dx>=dst_width||
        dy>=dst_height)
        {
            return;
        }

    int idx=dy*dst_width+dx;

//判断是否属于pad区域
    if (dx<pad_x||dy>=bottom||dx>=right||dy<pad_y)
    {
        dst[idx]=gray_val;
        dst[area+idx]=gray_val;
        dst[2*area+idx]=gray_val;
    }
    else
    {
//计算出原图中的位置，然后bgr转化成rgb
        float src_dx=(dx-pax_center)*x_scale-0.5f;
        float src_dy=(dy-pay_center)*y_scale-0.5f;

        int x0=floorf(src_dx);
        int y0=floorf(src_dy);

        int x1=x0+1;
        int y1=y0+1;

        float w00=(x1-src_dx)*(y1-src_dy);
        float w01=(src_dx-x0)*(y1-src_dy);
        float w10=(x1-src_dx)*(src_dy-y0);
        float w11=(src_dx-x0)*(src_dy-y0);
//clamp
        x0=max(0,x0);
        x1=min(x1,src_width-1);
        y0=max(0,y0);
        y1=min(y1,src_height-1);

        float r=loadBGR(src_tex,x0,y0,2)*w00+
                loadBGR(src_tex,x1,y0,2)*w01+
                loadBGR(src_tex,x0,y1,2)*w10+
                loadBGR(src_tex,x1,y1,2)*w11;

        float g=loadBGR(src_tex,x0,y0,1)*w00+
                loadBGR(src_tex,x1,y0,1)*w01+
                loadBGR(src_tex,x0,y1,1)*w10+
                loadBGR(src_tex,x1,y1,1)*w11;

        float b=loadBGR(src_tex,x0,y0,0)*w00+
                loadBGR(src_tex,x1,y0,0)*w01+
                loadBGR(src_tex,x0,y1,0)*w10+
                loadBGR(src_tex,x1,y1,0)*w11;

        int ri=__float2int_rn(r);
        int gi=__float2int_rn(g);
        int bi=__float2int_rn(b);

        ri=max(0,min(255,ri));
        gi=max(0,min(255,gi));
        bi=max(0,min(255,bi));

        dst[idx]=ri*stand;
        dst[area+idx]=gi*stand;
        dst[2*area+idx]=bi*stand;
    }
    
};

bool cudaPreprocessV2(    
    cudaTextureObject_t src_tex,
    int src_width,
    int src_height,
    int src_stride,

    float* device_dst,
    int dst_width,
    int dst_height,

    const letterboxmeta& meta,
    cudaStream_t stream
){
    dim3 block(32,8);

    dim3 grid(
        (dst_width+block.x-1)/block.x,
        (dst_height+block.y-1)/block.y
    );

    int right=meta.pad_x+meta.resized_width;//可移出
    int bottom=meta.pad_y+meta.resized_height;//可移出

    int area=dst_height*dst_width;//可移出
    float pax_center=meta.pad_x-0.5f;
    float x_scale=src_width*1.0f/meta.resized_width;

    float pay_center=meta.pad_y-0.5f;
    float y_scale=src_height*1.0f/meta.resized_height;

    float stand=1.0f/255.0f;

    float gray_val = 114.0f / 255.0f;

    preprocessKernalV2<<<grid,block,0,stream>>>(
        src_tex,
        src_width,
        src_height,

        device_dst,
        dst_width,
        dst_height,

        meta.pad_x,
        meta.pad_y,

        right,
        bottom,
        area,
        pax_center,
        x_scale,
        pay_center,
        y_scale,
        stand,
        gray_val
    );

    cudaError_t err=
        cudaGetLastError();

    return err==cudaSuccess;

};

//取值时使用texture，bilinear也使用texture
__global__ void preprocessKernalV3(
    cudaTextureObject_t src_tex,

    float* __restrict__ dst,
    int dst_width,
    int dst_height,

    int pad_x,
    int pad_y,
    int right,
    int bottom,
    int area,
    float pax_center,
    float x_scale,
    float pay_center,
    float y_scale,
    float gray_val
)
{

//dx和dy是当前thread在grid全局里的位置
    int dx=
        blockIdx.x*blockDim.x
        +threadIdx.x;
    int dy=
        blockIdx.y*blockDim.y
        +threadIdx.y;    
    if(dx>=dst_width||
        dy>=dst_height)
        {
            return;
        }

    int idx=dy*dst_width+dx;

//判断是否属于pad区域
    if (dx<pad_x||dy>=bottom||dx>=right||dy<pad_y)
    {
        dst[idx]=gray_val;
        dst[area+idx]=gray_val;
        dst[2*area+idx]=gray_val;
    }
    else
    {
//计算出原图中的位置，然后bgr转化成rgb
        float src_dx=(dx-pax_center)*x_scale;
        float src_dy=(dy-pay_center)*y_scale;

        float4 bgra=
            tex2D<float4>(
                src_tex,
                src_dx,
                src_dy
            );

        dst[idx]=bgra.z;
        dst[area+idx]=bgra.y;
        dst[2*area+idx]=bgra.x;
    }
    
};

bool cudaPreprocessV3(    
    cudaTextureObject_t src_tex,
    int src_width,
    int src_height,
    int src_stride,

    float* device_dst,
    int dst_width,
    int dst_height,

    const letterboxmeta& meta,
    cudaStream_t stream
){
    dim3 block(32,8);

    dim3 grid(
        (dst_width+block.x-1)/block.x,
        (dst_height+block.y-1)/block.y
    );

    int right=meta.pad_x+meta.resized_width;//可移出
    int bottom=meta.pad_y+meta.resized_height;//可移出

    int area=dst_height*dst_width;//可移出
    float pax_center=meta.pad_x-0.5f;
    float x_scale=src_width*1.0f/meta.resized_width;

    float pay_center=meta.pad_y-0.5f;
    float y_scale=src_height*1.0f/meta.resized_height;

    float gray_val = 114.0f / 255.0f;

    preprocessKernalV3<<<grid,block,0,stream>>>(
        src_tex,

        device_dst,
        dst_width,
        dst_height,

        meta.pad_x,
        meta.pad_y,

        right,
        bottom,
        area,
        pax_center,
        x_scale,
        pay_center,
        y_scale,
        gray_val
    );

    cudaError_t err=
        cudaGetLastError();
    
    if(err!=cudaSuccess){
        std::cerr
            <<"kernel launch error: "
            <<cudaGetErrorString(err)
            <<std::endl;
        return false;
    }

    return err==cudaSuccess;

};


__device__ __forceinline__
void preprocessSignalPixel(
    const uint8_t* __restrict__ src,
    int src_width,
    int src_height,
    int src_stride,

    float* __restrict__ dst,
    int dst_width,
    int dst_height,

    int area,
    int dy,
    int dx,

    float pax_center,
    float x_scale,
    float pay_center,
    float y_scale,

    float stand
)
{

        int idx=dy*dst_width+dx;
//计算出原图中的位置，然后bgr转化成rgb
        float src_dx=(dx-pax_center)*x_scale-0.5f;
        float src_dy=(dy-pay_center)*y_scale-0.5f;
//四个顶点对应的坐标
        int x0=floorf(src_dx);
        int y0=floorf(src_dy);

        int x1=x0+1;
        int y1=y0+1;
//四个顶点对应的权重
        float w00=(x1-src_dx)*(y1-src_dy);
        float w01=(src_dx-x0)*(y1-src_dy);
        float w10=(x1-src_dx)*(src_dy-y0);
        float w11=(src_dx-x0)*(src_dy-y0);
//clamp
        x0=max(0,x0);
        x1=min(x1,src_width-1);
        y0=max(0,y0);
        y1=min(y1,src_height-1);
//四个顶点对应的存储位置
        int index00=y0*src_stride+x0*3;
        int index01=y0*src_stride+x1*3;
        int index10=y1*src_stride+x0*3;
        int index11=y1*src_stride+x1*3;


        float r=src[index00+2]*w00+
                src[index01+2]*w01+
                src[index10+2]*w10+
                src[index11+2]*w11;

        float g=src[index00+1]*w00+
                src[index01+1]*w01+
                src[index10+1]*w10+
                src[index11+1]*w11;

        float b=src[index00]*w00+
                src[index01]*w01+
                src[index10]*w10+
                src[index11]*w11;

        int ri=__float2int_rn(r);
        int gi=__float2int_rn(g);
        int bi=__float2int_rn(b);

        ri=max(0,min(255,ri));
        gi=max(0,min(255,gi));
        bi=max(0,min(255,bi));

        dst[idx]=ri*stand;
        dst[area+idx]=gi*stand;
        dst[2*area+idx]=bi*stand;
};

//在v1的基础上改进，每个thread处理两个pixel
__global__ void preprocessKernalV4(
    const uint8_t* __restrict__ src,
    int src_width,
    int src_height,
    int src_stride,

    float* __restrict__ dst,
    int dst_width,
    int dst_height,

    int pad_x,
    int pad_y,
    int right,
    int bottom,
    int area,
    float pax_center,
    float x_scale,
    float pay_center,
    float y_scale,
    float stand,
    float gray_val
)
{

//dx和dy是当前thread在grid全局里的位置
    int tile_x=
        blockIdx.x*(blockDim.x*2);
    int dy=
        blockIdx.y*blockDim.y+threadIdx.y;    
    int dx0=tile_x+threadIdx.x;
    int dx1=dx0+blockDim.x;

    if(dx0>=dst_width||
        dy>=dst_height)
        {
            return;
        }

    int idx0=dy*dst_width+dx0;
    int idx1=dy*dst_width+dx1;
    bool has1=dx1<dst_width;

//判断是否属于pad区域
    if (dy>=bottom||dy<pad_y)
    {
        dst[idx0]=gray_val;
        dst[area+idx0]=gray_val;
        dst[2*area+idx0]=gray_val;

        if(has1)
        {
            dst[idx1]=gray_val;
            dst[area+idx1]=gray_val;
            dst[2*area+idx1]=gray_val;            
        }

        return;
    }

    bool valid0=
        dx0>=pad_x&&dx0<right;
    
    bool valid1=
        dx1>=pad_x&&dx1<right;

    if(!valid0)
    {
        dst[idx0]=gray_val;
        dst[area+idx0]=gray_val;
        dst[2*area+idx0]=gray_val;
    }

    if(has1&&!valid1)
    {
        dst[idx1]=gray_val;
        dst[area+idx1]=gray_val;
        dst[2*area+idx1]=gray_val;
    }

    if(!(valid0&&valid1))
    {
        if(valid0)
        {
            preprocessSignalPixel(
            src,
            src_width,
            src_height,
            src_stride,

            dst,
            dst_width,
            dst_height,

            area,
            dy,
            dx0,

            pax_center,
            x_scale,
            pay_center,
            y_scale,

            stand
         );
        }
        if(has1&&valid1)
        {
            preprocessSignalPixel(
            src,
            src_width,
            src_height,
            src_stride,

            dst,
            dst_width,
            dst_height,

            area,
            dy,
            dx1,

            pax_center,
            x_scale,
            pay_center,
            y_scale,

            stand
         );            
        }
        return;
    }

//y方向完全共享
        float src_dy=(dy-pay_center)*y_scale-0.5f;
        int y0=floorf(src_dy);
        int y1=y0+1;

        int row0=y0*src_stride;
        int row1=y1*src_stride;

        float dst_y0=src_dy-y0;
        float dst_y1=y1-src_dy;
        
        y0=max(0,y0);
        y1=min(y1,src_height-1);
//dx0对应的坐标
        float src_dx_0=(dx0-pax_center)*x_scale-0.5f;
        int x0_0=floorf(src_dx_0);
        int x1_0=x0_0+1;


        float src_dx_1=(dx1-pax_center)*x_scale-0.5f;
        int x0_1=floorf(src_dx_1);
        int x1_1=x0_1+1;

//dx0对应的权重
        float w00_0=(x1_0-src_dx_0)*dst_y1;
        float w01_0=(src_dx_0-x0_0)*dst_y1;
        float w10_0=(x1_0-src_dx_0)*dst_y0;
        float w11_0=(src_dx_0-x0_0)*dst_y0;

        float w00_1=(x1_1-src_dx_1)*dst_y1;
        float w01_1=(src_dx_1-x0_1)*dst_y1;
        float w10_1=(x1_1-src_dx_1)*dst_y0;
        float w11_1=(src_dx_1-x0_1)*dst_y0;
//clamp
        x0_0=max(0,x0_0);
        x1_0=min(x1_0,src_width-1);

        x0_1=max(0,x0_1);
        x1_1=min(x1_1,src_width-1);

//四个顶点对应的存储位置
        int index00_0=row0+x0_0*3;
        int index01_0=row0+x1_0*3;
        int index10_0=row1+x0_0*3;
        int index11_0=row1+x1_0*3;

        int index00_1=row0+x0_1*3;
        int index01_1=row0+x1_1*3;
        int index10_1=row1+x0_1*3;
        int index11_1=row1+x1_1*3;

        float p0_00=src[index00_0+2];
        float p0_01=src[index01_0+2];
        float p0_10=src[index10_0+2];
        float p0_11=src[index11_0+2];
        float p1_01=src[index01_1+2];
        float p1_10=src[index10_1+2];
        float p1_11=src[index11_1+2];
        float p1_00=src[index00_1+2];

        float r0= p0_00*w00_0+
                 p0_01*w01_0+
                 p0_10*w10_0+
                 p0_11*w11_0;

        float r1= p1_00*w00_1+
                 p1_01*w01_1+
                 p1_10*w10_1+
                 p1_11*w11_1;


        p0_01=src[index01_0+1];
        p0_10=src[index10_0+1];
        p0_11=src[index11_0+1];
        p1_01=src[index01_1+1];
        p1_10=src[index10_1+1];
        p1_11=src[index11_1+1];
        p0_00=src[index00_0+1];
        p1_00=src[index00_1+1];

        float g0= p0_00*w00_0+
                 p0_01*w01_0+
                 p0_10*w10_0+
                 p0_11*w11_0;

        float g1= p1_00*w00_1+
                 p1_01*w01_1+
                 p1_10*w10_1+
                 p1_11*w11_1;

        p0_01=src[index01_0];
        p0_10=src[index10_0];
        p0_11=src[index11_0];
        p1_01=src[index01_1];
        p1_10=src[index10_1];
        p1_11=src[index11_1];
        p0_00=src[index00_0];
        p1_00=src[index00_1];

        float b0= p0_00*w00_0+
                 p0_01*w01_0+
                 p0_10*w10_0+
                 p0_11*w11_0;

        float b1= p1_00*w00_1+
                 p1_01*w01_1+
                 p1_10*w10_1+
                 p1_11*w11_1;

        int ri0=__float2int_rn(r0);
        int gi0=__float2int_rn(g0);
        int bi0=__float2int_rn(b0);

        ri0=max(0,min(255,ri0));
        gi0=max(0,min(255,gi0));
        bi0=max(0,min(255,bi0));

        int ri1=__float2int_rn(r1);
        int gi1=__float2int_rn(g1);
        int bi1=__float2int_rn(b1);

        ri1=max(0,min(255,ri1));
        gi1=max(0,min(255,gi1));
        bi1=max(0,min(255,bi1));

        dst[idx0]=ri0*stand;
        dst[area+idx0]=gi0*stand;
        dst[2*area+idx0]=bi0*stand;

        dst[idx1]=ri1*stand;
        dst[area+idx1]=gi1*stand;
        dst[2*area+idx1]=bi1*stand;
    
};


bool cudaPreprocessV4(    
    const uint8_t*device_src,
    int src_width,
    int src_height,
    int src_stride,

    float* device_dst,
    int dst_width,
    int dst_height,

    const letterboxmeta& meta,
    cudaStream_t stream
){
    dim3 block(32,8);

    dim3 grid(
        (dst_width+block.x*2-1)/(block.x*2),
        (dst_height+block.y-1)/block.y
    );

    int right=meta.pad_x+meta.resized_width;//可移出
    int bottom=meta.pad_y+meta.resized_height;//可移出

    int area=dst_height*dst_width;//可移出
    float pax_center=meta.pad_x-0.5f;
    float x_scale=src_width*1.0f/meta.resized_width;

    float pay_center=meta.pad_y-0.5f;
    float y_scale=src_height*1.0f/meta.resized_height;

    float stand=1.0f/255.0f;

    float gray_val = 114.0f / 255.0f;

    preprocessKernalV4<<<grid,block,0,stream>>>(
        device_src,
        src_width,
        src_height,
        src_stride,

        device_dst,
        dst_width,
        dst_height,

        meta.pad_x,
        meta.pad_y,

        right,
        bottom,
        area,
        pax_center,
        x_scale,
        pay_center,
        y_scale,
        stand,
        gray_val
    );

    cudaError_t err=
        cudaGetLastError();

    return err==cudaSuccess;

};




__global__ void preprocessKernalV5(
    const uchar4* __restrict__ src,
    size_t src_pitch,

    int src_width,
    int src_height,

    float* __restrict__ dst,
    int dst_width,
    int dst_height,

    int pad_x,
    int pad_y,
    int right,
    int bottom,
    int area,
    float pax_center,
    float x_scale,
    float pay_center,
    float y_scale,
    float stand,
    float gray_val
)
{

//dx和dy是当前thread在grid全局里的位置
    int dx=
        blockIdx.x*blockDim.x+threadIdx.x;
    int dy=
        blockIdx.y*blockDim.y+threadIdx.y;    
    if(dx>=dst_width||
        dy>=dst_height)
        {
            return;
        }

    int idx=dy*dst_width+dx;

//判断是否属于pad区域
    if (dx<pad_x||dy>=bottom||dx>=right||dy<pad_y)
    {
        dst[idx]=gray_val;
        dst[area+idx]=gray_val;
        dst[2*area+idx]=gray_val;
    }
    else
    {
//计算出原图中的位置，然后bgr转化成rgb
        float src_dx=(dx-pax_center)*x_scale-0.5f;
        float src_dy=(dy-pay_center)*y_scale-0.5f;
//四个顶点对应的坐标
        int x0=floorf(src_dx);
        int y0=floorf(src_dy);

        int x1=x0+1;
        int y1=y0+1;
//四个顶点对应的权重
        float w00=(x1-src_dx)*(y1-src_dy);
        float w01=(src_dx-x0)*(y1-src_dy);
        float w10=(x1-src_dx)*(src_dy-y0);
        float w11=(src_dx-x0)*(src_dy-y0);
//clamp
        x0=max(0,x0);
        x1=min(x1,src_width-1);
        y0=max(0,y0);
        y1=min(y1,src_height-1);
//四个顶点对应的存储位置
        const uchar4* row0=
            reinterpret_cast<const uchar4*>(
                reinterpret_cast<const uint8_t*>(src)
                + static_cast<size_t>(y0)*src_pitch
            );

        const uchar4* row1=
            reinterpret_cast<const uchar4*>(
                reinterpret_cast<const uint8_t*>(src)
                + static_cast<size_t>(y1)*src_pitch
            );
        
        uchar4 p00=row0[x0];
        uchar4 p01=row0[x1];
        uchar4 p10=row1[x0];
        uchar4 p11=row1[x1];
        
        float r=static_cast<float>(p00.z)*w00+
                static_cast<float>(p01.z)*w01+
                static_cast<float>(p10.z)*w10+
                static_cast<float>(p11.z)*w11;

        float g=static_cast<float>(p00.y)*w00+
                static_cast<float>(p01.y)*w01+
                static_cast<float>(p10.y)*w10+
                static_cast<float>(p11.y)*w11;


        float b=static_cast<float>(p00.x)*w00+
                static_cast<float>(p01.x)*w01+
                static_cast<float>(p10.x)*w10+
                static_cast<float>(p11.x)*w11;

        int ri=__float2int_rn(r);
        int gi=__float2int_rn(g);
        int bi=__float2int_rn(b);

        ri=max(0,min(255,ri));
        gi=max(0,min(255,gi));
        bi=max(0,min(255,bi));

        dst[idx]=ri*stand;
        dst[area+idx]=gi*stand;
        dst[2*area+idx]=bi*stand;
    }
    
};


bool cudaPreprocessV5(    
    const uchar4*device_src,
    int src_pitch,

    int src_width,
    int src_height,

    float* device_dst,
    int dst_width,
    int dst_height,

    const letterboxmeta& meta,
    cudaStream_t stream
){
    dim3 block(32,8);

    dim3 grid(
        (dst_width+block.x-1)/block.x,
        (dst_height+block.y-1)/block.y
    );

    int right=meta.pad_x+meta.resized_width;//可移出
    int bottom=meta.pad_y+meta.resized_height;//可移出

    int area=dst_height*dst_width;//可移出
    float pax_center=meta.pad_x-0.5f;
    float x_scale=src_width*1.0f/meta.resized_width;

    float pay_center=meta.pad_y-0.5f;
    float y_scale=src_height*1.0f/meta.resized_height;

    float stand=1.0f/255.0f;

    float gray_val = 114.0f / 255.0f;

    preprocessKernalV5<<<grid,block,0,stream>>>(
        device_src,
        src_pitch,

        src_width,
        src_height,

        device_dst,
        dst_width,
        dst_height,

        meta.pad_x,
        meta.pad_y,

        right,
        bottom,
        area,
        pax_center,
        x_scale,
        pay_center,
        y_scale,
        stand,
        gray_val
    );

    cudaError_t err=
        cudaGetLastError();

    return err==cudaSuccess;

};



__global__ void preprocessKernalV6(
    const uint8_t* __restrict__ src,
    int src_width,
    int src_height,
    int src_stride,

    float* __restrict__ dst,
    int dst_width,
    int dst_height,

    int pad_x,
    int pad_y,
    int right,
    int bottom,
    int area,

    float pax_center,
    float x_scale,
    float pay_center,
    float y_scale,
    
    float stand,
    float gray_val
)
{

//dx和dy是当前thread在grid全局里的位置
    int dx=
        blockIdx.x*blockDim.x+threadIdx.x;
    int dy=
        blockIdx.y*blockDim.y+threadIdx.y;    
    if(dx>=dst_width||
        dy>=dst_height)
        {
            return;
        }

    int idx=dy*dst_width+dx;
    int oc=blockIdx.z;

//判断是否属于pad区域
    if (dx<pad_x||dy>=bottom||dx>=right||dy<pad_y)
    {
        dst[oc*area+idx]=gray_val;
    }
    else
    {
//计算出原图中的位置，然后bgr转化成rgb
        float src_dx=(dx-pax_center)*x_scale-0.5f;
        float src_dy=(dy-pay_center)*y_scale-0.5f;
//四个顶点对应的坐标
        int x0=floorf(src_dx);
        int y0=floorf(src_dy);

        int x1=x0+1;
        int y1=y0+1;
//四个顶点对应的权重
        float w00=(x1-src_dx)*(y1-src_dy);
        float w01=(src_dx-x0)*(y1-src_dy);
        float w10=(x1-src_dx)*(src_dy-y0);
        float w11=(src_dx-x0)*(src_dy-y0);
//clamp
        x0=max(0,x0);
        x1=min(x1,src_width-1);
        y0=max(0,y0);
        y1=min(y1,src_height-1);
//四个顶点对应的存储位置
        int index00=y0*src_stride+x0*3;
        int index01=y0*src_stride+x1*3;
        int index10=y1*src_stride+x0*3;
        int index11=y1*src_stride+x1*3;

        int ic=2-oc;
        float color=src[index00+ic]*w00+
                src[index01+ic]*w01+
                src[index10+ic]*w10+
                src[index11+ic]*w11;

        int colori=__float2int_rn(color);

        colori=max(0,min(255,colori));

        dst[oc*area+idx]=colori*stand;
    }
    
};


bool cudaPreprocessV6(    
    const uint8_t*device_src,
    int src_width,
    int src_height,
    int src_stride,

    float* device_dst,
    int dst_width,
    int dst_height,

    const letterboxmeta& meta,
    cudaStream_t stream
){
    dim3 block(32,8);

    dim3 grid(
        (dst_width+block.x-1)/block.x,
        (dst_height+block.y-1)/block.y,
        3
    );

    int right=meta.pad_x+meta.resized_width;//可移出
    int bottom=meta.pad_y+meta.resized_height;//可移出

    int area=dst_height*dst_width;//可移出
    float pax_center=meta.pad_x-0.5f;
    float x_scale=src_width*1.0f/meta.resized_width;

    float pay_center=meta.pad_y-0.5f;
    float y_scale=src_height*1.0f/meta.resized_height;

    float stand=1.0f/255.0f;

    float gray_val = 114.0f / 255.0f;

    preprocessKernalV6<<<grid,block,0,stream>>>(
        device_src,
        src_width,
        src_height,
        src_stride,

        device_dst,
        dst_width,
        dst_height,

        meta.pad_x,
        meta.pad_y,

        right,
        bottom,
        area,
        pax_center,
        x_scale,
        pay_center,
        y_scale,
        stand,
        gray_val
    );

    cudaError_t err=
        cudaGetLastError();

    return err==cudaSuccess;

};