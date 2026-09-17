# YOLOv8n TensorRT Video Pipeline

面向 NVIDIA Jetson AGX Orin 的 C++/CUDA 视频目标检测项目。通过零拷贝输入、GPU 前后处理、异步流水线和 CUDA Graph，最终版本吞吐达到约 **463 FPS**。

`main` 使用 GPU decode/filter + CUDA Graph。三个版本标签保留 CPU 后处理、GPU 后处理和 Graph 优化阶段，便于对比。

## Features

- **融合 CUDA 前处理**：完成双线性缩放、letterbox、BGR→RGB、归一化和 CHW 转换。
- **零拷贝输入**：通过 mapped host memory 直接读取视频帧。
- **异步流水线**：采集、GPU 提交、CPU 后处理三个线程，配合双帧槽、双 CUDA Stream 和 Event 同步。
- **GPU 后处理**：CUDA 完成 YOLO decode/filter，CPU 执行 class-aware NMS 和坐标还原。
- **CUDA Graph**：每个帧槽独立捕获 TensorRT、GPU decode/filter 和 D2H，逐帧 replay。
- **性能分析**：分阶段计时、NVTX 标记，以及 Nsight Systems / Compute 实验记录。

## Benchmark

| Version | 主要变化 | Throughput |
| --- | --- | ---: |
| [`v1-cpu-post`](https://github.com/hk1455/yolo-tensorrt-deploy/tree/v1-cpu-post) | CPU YOLO decode/filter | 253.708 FPS |
| [`v2-gpu-post`](https://github.com/hk1455/yolo-tensorrt-deploy/tree/v2-gpu-post) | CUDA decode/filter | 366.332 FPS |
| [`v3-cuda-graph`](https://github.com/hk1455/yolo-tensorrt-deploy/tree/v3-cuda-graph) | CUDA Graph replay | ~463 FPS |

配置：Jetson AGX Orin、YOLOv8n、TensorRT FP16、batch=1、模型输入 640×640。

吞吐统计视频处理循环，包含视频读取/解码、前处理、推理、decode/filter、NMS 和坐标还原；不包含模型构建、初始化、预热、画框及结果视频编码/写入。FPS 表示流水线吞吐，不是单帧延迟。

详细数据与测试口径见 [Benchmark notes](docs/benchmark.md)，优化过程见 [GPU 后处理与 CUDA Graph](results/ncu/day27_plus_summary.md)。

## Build and run

环境：JetPack 6.2.1、CUDA 12.6、TensorRT 10.3、OpenCV 4.8、GCC 11.4、CMake 3.27.9。需安装 CUDA、TensorRT、OpenCV 开发包及 NVTX 头文件。

在 Jetson 上执行：

```bash
git clone https://github.com/hk1455/yolo-tensorrt-deploy.git
cd yolo-tensorrt-deploy

cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release
cmake --build build-local --target build_engine video_detect -j"$(nproc)"

# 在目标设备生成 TensorRT engine，首次运行时执行
./build-local/build_engine
./build-local/video_detect
```

程序默认读取 `assets/island.mp4`，使用 `models/yolov8n_cpp_fp16_profile.engine`，并打印处理帧数、总耗时及 FPS。源视频尺寸从首帧读取，视频全程应保持固定分辨率。更换视频可修改 `apps/video_detect.cpp` 中的 `video_path`。

TensorRT engine 需在兼容的目标 GPU / 软件环境中构建；仓库不包含 engine 和编译产物。上述命令构建主视频程序与 engine builder，其他实验程序按需单独构建。

切换版本进行比较：

```bash
git switch --detach v1-cpu-post
cmake -S . -B build-v1 -DCMAKE_BUILD_TYPE=Release
cmake --build build-v1 --target build_engine video_detect -j"$(nproc)"
./build-v1/video_detect

# 返回最终实现
git switch main
```

v2/v3 分别使用 `v2-gpu-post / build-v2`、`v3-cuda-graph / build-v3`。比较时使用相同 engine、输入视频及设备功耗/锁频配置。

## Model assumptions

| Item | Configuration |
| --- | --- |
| Model | YOLOv8n，COCO 80 类 |
| Input | `images`，FP32 NCHW，视频入口为 `[1,3,640,640]` |
| Preprocess | RGB，letterbox padding=114，除以 255 |
| Engine profile | batch min/opt/max = 1/4/8，固定 640×640 |
| Raw output | `output0`，FP32 `[1,84,8400]` |
| Channels | 4 个 xywh + 80 个类别分数，无独立 objectness |
| Thresholds | confidence=0.20，NMS IoU=0.70 |

FP16 指 engine 内部计算精度，当前前后处理通路假设模型 I/O 为 FP32。GPU 版本回传候选计数和固定容量的 8400 个 `detection`，CPU 仅处理其中的有效候选。

## Output format

最终检测字段为 `class_id, confidence, x1, y1, x2, y2`，坐标为还原并裁剪到原图范围的像素坐标。

当前视频入口默认处于性能测试模式：检测结果在内存中处理，画框与视频写入关闭；`result.csv` 仅有表头，`result.mp4` 未写入帧。

## Correctness

- **数值对齐**：以 ONNX Runtime CPU FP32 为基准，比较相同输入下的 TensorRT raw output，覆盖 B1/B4 和 TF32 开关。见 [Numeric validation](docs/numeric_validation.md)。
- **前处理校验**：CUDA 前处理与 CPU 参考实现对比，并保留多种 kernel 实验。见 [CUDA preprocess](results/ncu/Day19%20CUDA%20Preprocess%20Profiling%20Summary.md)。
- **FP16 校验**：记录 raw output 误差和样例检测结果对比。见 [FP16 validation](results/ncu/day21_summary.md)。

以上为数值及样例级验证，不包含完整数据集的 mAP 评估。

## Code map

| Path | Purpose |
| --- | --- |
| `apps/video_detect.cpp` | 初始化、预热、线程启动和吞吐统计 |
| `src/fram_slot.cpp` | 采集、GPU 提交、CPU 后处理与 Graph |
| `src/cuda_preprocess.cu` | V0～V6 前处理实验，主流程采用 V1 |
| `src/cuda_postprocess.cu` | CUDA YOLO decode/filter |
| `src/trt_engine.cpp` | TensorRT runtime |
| `results/benchmark/`、`results/ncu/` | 测试数据和性能分析记录 |

版本整理说明见 [Version notes](docs/benchmark.md#version-notes)。`v0.2-cpu-golden` 保留早期单图基线。
