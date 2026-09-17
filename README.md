# YOLOv8n TensorRT Video Pipeline

在 Jetson AGX Orin 上部署 YOLOv8n，比较 CPU 后处理、GPU 后处理和 CUDA Graph 三个视频流水线版本。

## Versions / Benchmark

| Version | 主要变化 | Throughput | 记录状态 |
| --- | --- | --- | --- |
| `v1-cpu-post` | CPU YOLO decode/filter，无 Graph | 253.708 FPS | 已在 Jetson 上复测 |
| `v2-gpu-post` | CUDA decode/filter，无 Graph | 366.332 FPS | 已在 Jetson 上复测 |
| `v3-cuda-graph` | CUDA Graph replay | ~463 FPS | 最终版已在 Jetson 上稳定实测 |

当前代码基于 **v3-cuda-graph**；GitHub `main` 保持 GPU 后处理 + CUDA Graph 最终实现，并持续更新文档。

三个标签指向三个不同提交，按 CPU 后处理 → GPU 后处理 → CUDA Graph 顺序组织。
v1/v2 从保留的源码恢复后，已在 Jetson 上重新运行并完成吞吐测试；v3 是最初提供时就已在 Jetson 上稳定达到约 463 FPS 的 CUDA Graph 最终版。

2026-09-17 更新：v1/v2 复测结果如下，表中的 253.708 / 366.332 FPS 均为本次实测值：

| 执行路径 | 处理帧数 | 总耗时 | 原始记录（含测试源码） |
| --- | ---: | ---: | --- |
| CPU 后处理，无 Graph | 1697 | 6.6888 s | [CPU 复测](results/benchmark/retest-2026-09-17/cpu-post-no-graph.txt) |
| GPU 后处理，无 Graph | 1697 | 4.63241 s | [GPU 复测](results/benchmark/retest-2026-09-17/gpu-post-no-graph.txt) |

v1/v2 的 FPS 使用本次复测程序的原始打印值，原始记录包含测试源码。
v3 的约 463 FPS 是最终版在 Jetson 上稳定运行的实测结果，相关优化过程见 [CUDA Graph 优化记录](results/ncu/day27_plus_summary.md)。三个版本均有实际运行的吞吐结果。
当前代码关闭了画框和视频写入，CSV 仅写表头；吞吐不含检测结果的视频编码/磁盘写入。

## Shared pipeline

三个版本均使用：
- YOLOv8n、TensorRT FP16、batch=1、模型输入 640×640。
- 视频首帧决定源图像尺寸；输入视频应保持固定分辨率。
- Mapped host memory、CUDA preprocess V1（32×8 block）。
- Capture / GPU Worker / CPU Postprocess 三线程，两个 FrameSlot。
- 前处理、推理两个 CUDA Stream，以 CUDA Event 同步。
- confidence=0.20、IoU=0.70；CPU 执行 class-aware NMS 和坐标还原。

| Tag | Decode/filter | D2H 内容 | TensorRT 提交 |
| --- | --- | --- | --- |
| v1-cpu-post | CPU | 完整 FP32 raw output | 每帧 enqueueV3 |
| v2-gpu-post | GPU | 候选计数 + 固定容量候选数组 | 每帧 enqueueV3 |
| v3-cuda-graph | GPU | 候选计数 + 固定容量候选数组 | 每个 slot 独立 graphExec，逐帧 replay |

Graph 包含 TensorRT、计数清零、GPU decode/filter 和 D2H，前处理仍在独立 Stream。
GPU 版本固定回传 8400 个 detection 的容量，并非按有效框数量动态拷贝。

## Build instructions

已记录环境：Jetson AGX Orin / JetPack 6.2.1 / CUDA 12.6 / TensorRT 10.3 /
OpenCV 4.8 / GCC 11.4 / CMake 3.27.9。需要 CUDA、TensorRT、OpenCV 开发包及 NVTX 头文件。

以下命令在 Jetson 的项目根目录执行。Windows 不能直接运行 Jetson 的 ELF 或复用其 engine。

```bash
# 可选：切换版本；返回最终版用 git switch main
git switch --detach v1-cpu-post

# 每个版本使用独立构建目录；改为当前标签名
cmake -S . -B build-v1 -DCMAKE_BUILD_TYPE=Release
cmake --build build-v1 --target build_engine video_detect -j"$(nproc)"

# 在目标 Jetson 上生成模型；三个版本可使用同一份兼容 engine
./build-v1/build_engine
./build-v1/video_detect
```

v2/v3 对应改用 `v2-gpu-post / build-v2` 和 `v3-cuda-graph / build-v3`。
当前构建命令只构建视频入口和 engine builder；其他实验入口需单独检查兼容性。

默认视频：`assets/island.mp4`。
默认 engine：`models/yolov8n_cpp_fp16_profile.engine`。
换视频可修改 `apps/video_detect.cpp` 的 `video_path`。
engine profile 为 batch 1/4/8、固定 640×640，视频入口使用 batch=1。
程序打印处理帧数、耗时和 FPS。

## Model assumptions / Output format

- 输入：`images`，FP32 NCHW，RGB，letterbox padding=114，除以 255。
- 输出：`output0`，FP32 `[1,84,8400]`；4 个 xywh 通道 + 80 类分数，无独立 objectness。
- FP16 指 engine 内部允许 FP16 计算，当前数据通路仍假设 I/O 为 FP32。
- 最终 detection：`class_id, confidence, x1, y1, x2, y2`，坐标为原图像素坐标。
- 当前视频入口用于性能测试，不输出逐帧检测记录；`result.csv` 仅表头，`result.mp4` 未写入帧。

## Correctness and reconstruction notes

已有 [FP32 raw-output 校验](docs/numeric_validation.md)、
[CUDA 前处理实验](results/ncu/Day19%20CUDA%20Preprocess%20Profiling%20Summary.md) 和
[FP16 对比记录](results/ncu/day21_summary.md)。上面的吞吐测试与这里的数值、检测结果验证分别记录。

恢复时保留视频实际尺寸读取和单图输出 buffer 分配，并修正：
- GPU 预热前初始化输入、清零 detection counter，避免未初始化数据/计数导致越界。
- 无 Graph 版本拷贝候选数组使用 `8400 * sizeof(detection)`，修正旧注释中的 `sizeof(float)`。
- Graph 初始化及提交失败时终止，避免继续使用无效结果。

运行及性能状态：v1/v2 已完成 Jetson 复测，v3 最终版已稳定实测约 463 FPS。
本次 v1/v2 复测附件记录了吞吐，未包含新的检测精度对比结果。
旧标签 `v0.2-cpu-golden` 保留原单图基线。

## Code map

- `apps/video_detect.cpp`：初始化、预热、线程启动与 FPS。
- `src/fram_slot.cpp`：采集、GPU 提交、CPU 后处理。
- `src/cuda_preprocess.cu`：V0～V6 前处理实验，视频主流程采用 V1。
- `src/cuda_postprocess.cu`：YOLO decode/filter。
- `src/trt_engine.cpp`：TensorRT runtime。
- `results/benchmark/`、`results/ncu/`：历史实验数据与总结。
