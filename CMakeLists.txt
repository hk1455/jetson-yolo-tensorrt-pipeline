cmake_minimum_required(VERSION 3.18)

project(yolo_tensorrt_deploy)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(
    build_engine 
    apps/build_engine.cpp
    src/engine_builder.cpp
)
target_include_directories(
    build_engine
    PRIVATE   
    /usr/local/cuda-12.6/targets/aarch64-linux/include
)
target_link_libraries(
    build_engine 
    PRIVATE nvinfer 
    nvonnxparser
    CUDA::cudart
)





add_executable(
    inspect_engine 
    apps/inspect_engine.cpp
    src/trt_engine.cpp
)
target_include_directories(
    inspect_engine
    PRIVATE   
    /usr/local/cuda-12.6/targets/aarch64-linux/include
)
target_link_libraries(
    inspect_engine 
    PRIVATE nvinfer 
    nvinfer
    CUDA::cudart
)




add_executable(
    runtime_engine 
    apps/run_engine.cpp
    src/trt_engine.cpp  
)
target_include_directories(
    runtime_engine
    PRIVATE   
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/local/cuda-12.6/targets/aarch64-linux/include
)
target_link_libraries(
    runtime_engine 
    PRIVATE 
    nvinfer
    CUDA::cudart
)




add_library(
    preprocess 
    src/preprocess.cpp
)

target_include_directories(
    preprocess 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    preprocess 
    PUBLIC
         ${OpenCV_LIBS}
)

add_executable(
    test_preprocess 
    apps/test_preprocess.cpp
)

target_link_libraries(
    test_preprocess
    PRIVATE 
        preprocess
)


add_executable(
    postprocess 
    apps/postprocess_raw.cpp
    src/postprocess.cpp  
)
target_include_directories(
    postprocess
    PRIVATE   
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    postprocess
    PUBLIC
         ${OpenCV_LIBS}
)

add_executable(
    yolo_detect
    apps/yolo_detect.cpp
    src/trt_engine.cpp
    src/preprocess.cpp
    src/postprocess.cpp
)
target_include_directories(yolo_detect
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo_detect
    PRIVATE
        nvinfer
        CUDA::cudart
        ${OpenCV_LIBS}
)
