#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "preprocess.hpp"

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
);

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
);

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
);


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
);

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
);

bool cudaPreprocessV5(
    const uchar4*device_src,
    int dst_pitch,
    int src_width,
    int src_height,

    float* device_dst,
    int dst_width,
    int dst_height,

    const letterboxmeta& meta,
    cudaStream_t stream
);

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
);
