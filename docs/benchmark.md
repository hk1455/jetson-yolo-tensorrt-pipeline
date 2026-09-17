# Benchmark notes

## Measurements

| Version | Frames | Time | Throughput | Source |
| --- | ---: | ---: | ---: | --- |
| v1-cpu-post | 1697 | 6.6888 s | 253.708 FPS | [CPU record](../results/benchmark/retest-2026-09-17/cpu-post-no-graph.txt) |
| v2-gpu-post | 1697 | 4.63241 s | 366.332 FPS | [GPU record](../results/benchmark/retest-2026-09-17/gpu-post-no-graph.txt) |
| v3-cuda-graph | — | — | ~463 FPS | [Graph optimization](../results/ncu/day27_plus_summary.md) |

v1/v2 数据来自 2026-09-17 提供的 Jetson 运行记录，FPS 保留程序打印值；附件包含测试源码。v3 为项目作者确认可稳定达到约 463 FPS 的最终实现。

平台为 Jetson AGX Orin，YOLOv8n TensorRT FP16，batch=1，模型输入 640×640。运行记录没有完整保存每次测试的功耗与锁频状态；复现时应固定并记录这些条件。

## Timing scope

- 计入：视频读取/解码、CUDA 前处理、TensorRT、decode/filter、CPU NMS、坐标还原及流水线调度。
- 不计入：engine 构建、初始化、预热、画框和结果视频编码/写入。
- 计时实现：启动三个线程后记录开始时间，等待线程结束后统计 processed_frames / elapsed_seconds。因此存在少量线程启动计时偏差。
- FPS 是流水线吞吐，不能直接作为每帧端到端延迟。
- 当前画框和写视频代码关闭；CSV 只写表头。
- 吞吐附件不包含新的检测精度对比，数值与样例级验证见 README 的 Correctness 部分。

## Version notes

三个标签对应三个不同提交：
- v1-cpu-post：CPU decode/filter，完整 raw output 回传。
- v2-gpu-post：CUDA decode/filter，回传候选计数及固定容量候选数组。
- v3-cuda-graph：每个 slot 使用独立 CUDA Graph 执行 TensorRT、GPU 后处理和 D2H。

v1/v2 根据保留的源码恢复，并完成运行及吞吐测试；这些标签不是早期实验时保存的原始 Git 快照。测试附件未记录提交号，保存附件用于追溯实际测试源码。

整理时三个版本均保留视频首帧尺寸读取和单图输出 buffer 分配；同时补充预热输入初始化、GPU 检测计数器清零，修正无 Graph 候选回传长度为 8400 * sizeof(detection)，并检查 Graph 初始化/提交错误。

main 保留最终实现并持续更新文档，版本标签保留各自提交快照。
