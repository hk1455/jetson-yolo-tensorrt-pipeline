# Day25 Summary — 三线程流水线

## 实验配置

- 平台：Jetson AGX Orin
- 模型：YOLOv8n，TensorRT FP16
- Batch：1，输入尺寸：640×640
- FrameSlot：2 个
- CPU 线程：Capture / GPU Worker / Result Consumer
- CUDA Streams：Preprocess / Inference

## Pipeline

```text
free_slots
    ↓
Capture → frame_queue → GPU Worker → result_queue → CPU Postprocess
    ↑                                               │
    └──────────────── Slot 归还 ─────────────────────┘
```

`FrameSlot` 统一持有图像、GPU buffer、输出 buffer、metadata 和 CUDA events。Queue 只传递 `slot_index`，通过 ownership 转移避免线程间复制整帧数据。

GPU Worker 异步提交 preprocess、TensorRT 和 D2H；Result Consumer 等待 `output_ready` 后执行 decode、NMS、reverse letterbox，最后归还 Slot。

## 实测结果

| 版本 | 帧数 | 总耗时 | FPS |
|---|---:|---:|---:|
| Day24 — 双 Stream | 1000 | 5.602 s | 178.511 |
| **Day25 — 三线程** | **1697** | **6.577 s** | **258.004** |

**相对 Day24，吞吐量提升约 44.5%。**

## 结论

Day25 将 CPU postprocess 从 GPU submission 路径中拆分出来，使下面的 overlap 成为可能：

```text
TRT(N+1) || CPU Postprocess(N)
```

实测吞吐量明显提升，符合减少 CPU postprocess 阻塞的预期。检测结果 CSV 和画框视频也已验证正常。

**下一步：**使用相同帧数和测试条件进行严格对照，并测量 E2E latency 分布及实际 GPU/CPU overlap。