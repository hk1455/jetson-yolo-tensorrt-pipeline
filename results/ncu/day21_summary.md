# Day21 TensorRT FP16 Summary

## 1. Goal

Day21 的目标是将 Day20 已经跑通的 FP32 TensorRT pipeline 升级到 FP16，并完成：

- Correctness 验证
- FP32 vs FP16 Performance 对比
- TensorRT tactic / Tensor Core profiling
- 最终是否保留 FP16 的结论

本阶段保持以下条件不变：

```text
Batch: B1
Input: 1x3x640x640
CUDA Preprocess: V1
Input Path: Zero-copy
Postprocess: CPU
Benchmark Method: Same
```

唯一主要变量：

```text
TensorRT Precision
FP32 → FP16
```

---

## 2. Correctness

FP32 与 FP16 raw output：

```text
Shape: (84, 8400)

Mean Abs Error:   0.0053349
Median Abs Error: 3.90e-08
P99 Abs Error:    0.1326
Max Abs Error:    8.5623
```

较大的 absolute error 主要出现在 box channels。

虽然 raw tensor 存在 FP16 numerical drift，但最终 FP32 / FP16 Detection 截图对比中：

- Detection 数量基本一致
- Class 基本一致
- Bounding Box 基本一致
- Confidence 基本一致
- 未观察到明显 Detection-level regression

因此当前测试样本下：

```text
Detection-level Correctness: PASS
```

---

## 3. Performance

### FP32 Baseline

```text
Preprocess p50: ≈ 0.102 ms
TensorRT p50:   ≈ 4.083 ms
D2H p50:        ≈ 0.085 ms
Total p50:      ≈ 4.270 ms
```

### FP16

```text
Preprocess p50: 0.102 ms
TensorRT p50:   2.284 ms
D2H p50:        0.087 ms
Total p50:      2.472 ms
```

### Speedup

```text
TensorRT:
4.083 ms → 2.284 ms
Speedup ≈ 1.79×

Total:
4.270 ms → 2.472 ms
Speedup ≈ 1.73×
```

CUDA Preprocess 和 D2H 基本没有明显变化，说明主要性能收益来自 TensorRT FP16 inference。

---

## 4. Nsight Compute Profiling

### FP32 Sampled Conv Kernel

Kernel path：

```text
XMMA Implicit GEMM
FP32 / TF32
Tensor Core
```

主要指标：

```text
HMMA Active:     ≈ 83.46%
Kernel Duration: ≈ 304 us
```

Kernel name 中可以看到：

```text
f32f32_tf32f32_f32
tensor16x8x8
```

说明该 FP32 TensorRT kernel 实际使用的是 TF32 Tensor Core execution path。

### FP16 Sampled Conv Kernel

Kernel path：

```text
XMMA Implicit GEMM
FP16
Tensor Core
```

主要指标：

```text
HMMA Active:     ≈ 71.24%
Kernel Duration: ≈ 183 us
```

Kernel name 中可以看到：

```text
f16f16_f16f16_f16
tensor16x8x16
```

Nsight Compute 同时明确显示：

```text
Tensor is the highest-utilized pipeline
It executes 16-bit floating point tensor operations
```

因此可以确认 FP16 kernel 使用了 Tensor Core 相关执行路径。

---

## 5. Tactic Observation

FP32 和 FP16 使用了不同的 XMMA implicit-GEMM implementation。

例如：

```text
FP32:
tilesize128x256x32
warpsize2x4x1
tensor16x8x8

FP16:
tilesize256x128x32
warpsize4x2x1
tensor16x8x16
```

这说明 FP16 的性能提升并不是简单地把同一个 kernel 的 datatype 从 float 改成 half。

TensorRT 根据 precision 和硬件特性选择了不同的 optimized tactic。

同时，FP32 engine 本身也使用 TF32 Tensor Core，因此不能简单表述为：

```text
FP32 = CUDA Core
FP16 = Tensor Core
```

更准确的结论是：

```text
FP32:
TF32 Tensor Core tactic

FP16:
FP16 Tensor Core tactic
+
Different XMMA implementation
```

抽样 Conv kernel latency：

```text
304 us → 183 us
```

与整体 TensorRT latency：

```text
4.083 ms → 2.284 ms
```

的变化趋势一致。

---

## 6. Final Conclusion

Day21 完成了 TensorRT FP32 → FP16 的完整验证。

结果：

```text
Correctness:
PASS

TensorRT Speedup:
≈ 1.79×

E2E Speedup:
≈ 1.73×

Tensor Core Evidence:
Confirmed
```

Profiler 证明：

- FP32 sampled Conv 使用 TF32 Tensor Core
- FP16 sampled Conv 使用 FP16 Tensor Core
- FP32 / FP16 使用不同 XMMA implicit-GEMM tactics
- FP16 sampled kernel latency 明显下降

因此最终保留 FP16 TensorRT engine。

当前 pipeline：

```text
Mapped Host BGR Image
↓
Zero-copy GPU Access
↓
CUDA Preprocess V1
↓
TensorRT FP16
↓
Raw D2H
↓
CPU Postprocess
```

Day21 COMPLETE.

Next:

```text
Day22
Serial Execution
↓
Async / Stream / Synchronization / Overlap
```
