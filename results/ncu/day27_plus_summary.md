# Day27 以后优化总结

## 1. GPU 后处理迁移
- 将 YOLO 输出的 `decode/filter` 从 CPU 迁移到自定义 CUDA kernel。
- 使用 `atomicAdd` 压缩有效检测框，只回传检测结果，不再拷贝完整 `84×8400` 输出。
- CPU Postprocess 从约 **2.87 ms** 降到约 **0.05 ms**。
- 整体吞吐从约 **250 FPS 提升到 384 FPS**。

## 2. CUDA Graph 优化 TensorRT 提交
- Nsight Systems 发现 TensorRT 内存在大量短 kernel，kernel 之间有明显 host launch gap。
- 将 **TensorRT + GPU decode/filter + D2H** capture 到 CUDA Graph，每个 slot 使用独立 `graphExec`。
- preprocess 保持独立 stream，继续保留 `Pre(N+1) || Infer(N)` 重叠。
- Graph 后 kernel 基本连续，吞吐从约 **384 FPS 提升到 463 FPS**。

## 3. Graph 间隙定量分析
- 将 Nsight 报告导出为 SQLite，自动统计 Graph-to-Graph gap。
- 平均 Graph gap：**179.4 μs**。
- 进一步拆分为：
  - 等待下一帧 `Frame Pop`：**62.29 μs**
  - Frame ready → preprocess start：**31.40 μs**
  - 暴露的 preprocess kernel：**60.74 μs**
  - preprocess end → next Graph：**24.98 μs**
- 说明 CUDA Graph 后，瓶颈已转移到 **Capture / frame feeding + preprocess**。

## 4. Pipeline 深度与 Capture 实验
- slot 从 2 增加到 4 几乎无收益，说明问题不是 buffer depth 不足。
- OpenCV FFmpeg backend 中：
  - `grab()` 约 **80 μs**
  - `retrieve()` 约 **1.98 ms**
- 尝试 Jetson GStreamer `nvv4l2decoder` 后，由于 `NVMM → BGRx → BGR → cv::Mat` 的转换/拷贝开销，整体仅约 **150 FPS**，因此暂不采用。

## 当前结果
- Baseline：约 **250 FPS**
- GPU Postprocess：约 **384 FPS**
- CUDA Graph：约 **463 FPS**
- 当前主要限制：**Capture/Decode feeding 与 preprocess overlap**

整体形成了完整的优化闭环：

**Profiling → 定位瓶颈 → 提出假设 → 实现优化 → Nsight 验证 → 瓶颈迁移**
