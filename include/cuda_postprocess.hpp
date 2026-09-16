#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "postprocess.hpp"
#include "fram_slot.hpp"

bool launchDecodeFilter(
    const float*device_output, 
    detection* device_detections,
    int* device_det_detection,
    float confidence_threshold,
    cudaStream_t stream
);