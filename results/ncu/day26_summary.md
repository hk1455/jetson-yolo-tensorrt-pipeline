# Day26 Summary — Nsight Systems 流水线验证

## 目标

Day26 不再增加新功能，而是使用 **Nsight Systems + NVTX** 验证 Day24/25 的异步流水线是否按预期工作，并据此确定 Day27 的优化方向。

## 已验证的流水线行为

通过 Nsight Systems 时间线确认：

- `Preprocess(N+1)` 与 `TensorRT(N)` 在两个 CUDA Stream 上存在真实重叠。
- `CPU Postprocess(N-1)` 与 `TensorRT(N)` 存在真实重叠。
- Capture / GPU Worker / CPU Postprocess 三个线程已经实现解耦。
- `ExecutionContext::enqueue` / `Submit TensorRT` 是 CPU 侧提交范围，真正的 GPU execution 需要看 `CUDA HW (Orin)` 时间线。
- `Wait Output Ready` 很短，说明 GPU → CPU 的 event 同步本身不是主要阻塞点。

## 整段 Nsight 统计

典型平均耗时：

| 阶段 | 平均耗时 |
|---|---:|
| Submit TensorRT | 3.293 ms |
| TensorRT `ExecutionContext::enqueue` | 3.280 ms |
| CPU Postprocess | 2.872 ms |
| Queue Wait / Free Slot | 2.214 ms |
| Capture / Decode | 2.085 ms |
| Queue Wait / Result Pop | 1.336 ms |
| Queue Wait / Frame Pop | 0.942 ms |
| Wait Output Ready | 0.120 ms |

时间线上多次观察到：

```text
CPU Postprocess 结束
        ↓
Free Slot Wait 结束
        ↓
Capture/Decode
        ↓
Frame Pop Wait 结束
        ↓
Submit TensorRT
```

说明 CPU Postprocess 虽然与 GPU inference 有部分重叠，但仍会通过 **slot release → Capture → GPU next frame** 进入吞吐关键路径。

## Postprocess A/B 实验

CPU 后处理内部测得：

```text
process()            ≈ 2.846 ms
NMS                  ≈ 0.048 ms
Reverse Letterbox    ≈ 0.00057 ms
```

正常流水线吞吐约：

```text
≈ 250 FPS
```

注释掉 `process()` 后，多次测试稳定约：

```text
≈ 377 FPS
```

说明 `process()` 对整体吞吐有显著影响，是当前一个重要的 **critical-path contributor**。NMS 和坐标还原占比很小，不是优先优化对象。

## Day26 结论

Day24/25 设计的 overlap 已经在真实时间线上得到验证：

```text
Pre(N+1) || TensorRT(N)
CPU Post(N-1) || TensorRT(N)
```

但 overlap 并不代表 CPU Postprocess 已完全被隐藏。Nsight 时间线和 A/B 实验都表明，`process()` 的部分耗时仍暴露在吞吐关键路径上，并影响 FrameSlot 释放和下一帧提交。

## Day27 假设

优先优化 `process()`，方向是将 TensorRT 输出的 **decode / confidence filter / candidate generation / compaction** 从 CPU 移到 GPU。

第一版无需急着迁移 NMS 和 Reverse Letterbox；先验证 GPU `process()` 是否能够：

- 缩短 CPU Postprocess critical-path tail；
- 更早释放 FrameSlot；
- 减少 GPU Worker 的等待；
- 提升整体 FPS；
- 在 Nsight 时间线上减少 pipeline bubble。

Day26 至此可以收尾。
