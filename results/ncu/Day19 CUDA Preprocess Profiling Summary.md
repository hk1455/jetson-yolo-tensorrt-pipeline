# Day19 CUDA Preprocess Profiling Summary

## Baseline

最终基准版本为：

```text
V1
- BGR3
- 1 thread / output pixel
- Global Memory
- Software Bilinear
- Block: 32×8
- p50 ≈ 0.094 ms
```

Nsight Compute 显示主要瓶颈为：

```text
Long Scoreboard
+
Uncoalesced Source Loads
+
High Excessive Sectors
```

Source hotspot 主要集中在 bilinear 的 12 个 BGR source loads，而 CHW output store 基本规整。

---

## Optimization Experiments

| Version | Strategy | p50 | Result |
|---|---|---:|---|
| V1 | BGR3 + Global + Software Bilinear | ~0.094 ms | Best |
| V2 | Point Texture + Software Bilinear | ~0.124 ms | Texture Queue Saturation |
| V3 | BGRA4 + Hardware Bilinear Texture | ~0.133 ms | Long Scoreboard / Low L1 Hit |
| V4 | 2 Pixels / Thread | ~0.095 ms | Long Scoreboard ↓, LG Throttle ↑ |
| V5 | BGRA4 + uchar4 Global Load | ~0.096 ms | LG Pressure ↓, Long Scoreboard ↑ |

### V2

Texture 没有减少 fetch 数量，仍然需要大量 narrow texture fetch，最终出现：

```text
Texture Queue Saturation
```

性能明显下降。

### V3

Hardware Bilinear 减少了 arithmetic 和显式 fetch，但：

```text
L1/TEX Hit Rate ≈ 8.66%
Long Scoreboard ≈ 89%
```

texture fetch latency 过高，性能进一步下降。

### V4

Thread Coarsening 增加 ILP / MLP：

```text
Long Scoreboard
24.5 → 17.2 cycles
```

说明能够隐藏部分 memory latency。

但同时出现：

```text
LG Throttle
```

最终没有实际加速。

实验中曾出现约 `0.068 ms`，后确认是错误 source index 导致的 False Speedup。

### V5

使用 `uchar4` wide load 后：

```text
Total Sectors ↓ ~53%
Excessive Sectors ↓ ~61%
LG Throttle ↓
```

但：

```text
L1 Hit Rate ↓
L2 Hit Rate ↓
Long Scoreboard ↑
```

说明减少 memory instruction 并不代表 memory latency 一定降低。

---
### V6：参考 tensorrt-cpp-api 的 Channel-parallel 实现

V6 参考 `tensorrt-cpp-api` preprocess，将线程映射改为：

```text
1 thread
→ 1 output element/channel
```

Correctness PASS：

```text
Max Abs Error  ≈ 1/255
Mean Abs Error ≈ 5.4e-6
```

Benchmark：

```text
V1 p50 ≈ 0.094 ms
V6 p50 ≈ 0.215 ms
```

NCU：

```text
Long Scoreboard        ≈ 38.9 cycles / 88.1%
Excessive Sectors      ≈ 2.135 M
Excessive Ratio        ≈ 89%
Branch Instructions    ≈ 157 K
No Eligible            ≈ 75.7%
```

V6 的 `Excessive Sectors` 与 V1 基本相同，说明 channel-parallel mapping 没有改善 bilinear source access。

同时 R/G/B 三个 channel 分别由不同 thread 处理，使 source coordinate、boundary check 和 interpolation geometry 被重复计算。

因此 V6 更适合作为通用 fused preprocess 实现，而针对固定 3-channel YOLO workload，V1 的 `1 thread / pixel → 同时生成 RGB` 能共享更多计算，性能明显更好。

**Conclusion：Correctness PASS，Performance Rejected。**


## Final Conclusion

多个 profiler-driven experiments 都没有稳定击败 V1。

V1 虽然存在较高的 `Long Scoreboard` 和不规整的 BGR3 source access，但在：

```text
Memory Footprint
Cache Locality
Memory Instruction Pressure
Latency
```

之间取得了目前最好的平衡。

最终冻结：

```text
CUDA Preprocess: V1
Block: 32×8
p50 ≈ 0.094 ms
```

Day19 的核心结论是：

> 优化一个指标并不代表 kernel 一定更快，瓶颈经常只是发生迁移。最终性能必须由 Correctness、Benchmark 和 Re-profile 共同判断。

下一阶段进入 TensorRT Integration。