#include "cuda_postprocess.hpp"
#include "fram_slot.hpp"
#include<iostream>
#include<cuda_runtime.h>
#include<cstdint>

__global__
void decodeFilterKernel(
    const float* device_output,
    detection* detections,
    int* det_count,
    float confidence_threshold
){
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    if(i>=8400)
        return;
    
    float max_confidence=device_output[4*8400+i];

    int class_id=0;

    for(int j=1;j<80;j++)
    {
        float conf=device_output[(4+j)*8400+i];
        if(conf>max_confidence)
        {
            max_confidence=conf;
            class_id=j;
        }
    }

    if(max_confidence<confidence_threshold)
        return;

    int index=atomicAdd(det_count,1);

    float cx=device_output[i];
    float cy=device_output[8400+i];
    float w=device_output[2*8400+i];
    float h=device_output[3*8400+i];

    detection d;
    d.x1=cx-w*0.5f;
    d.y1=cy-h*0.5f;
    d.x2=cx+w*0.5f;
    d.y2=cy+h*0.5f;

    d.confidence=max_confidence;
    d.class_id=class_id;

    detections[index]=d;
}

bool launchDecodeFilter(
    const float*device_output, 
    detection* device_detections,
    int* device_det_detection,
    float confidence_threshold,
    cudaStream_t stream
){
    constexpr int NUM_CANDIDATES=8400;
    constexpr int THREADS=256;

    int block=(NUM_CANDIDATES+THREADS-1)/THREADS;

    decodeFilterKernel<<<block,THREADS,0,stream>>>(
        device_output,
        device_detections,
        device_det_detection,
        confidence_threshold
    );

    return cudaGetLastError()==cudaSuccess;
}
