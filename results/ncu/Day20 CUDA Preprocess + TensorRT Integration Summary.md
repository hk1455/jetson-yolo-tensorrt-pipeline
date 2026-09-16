# Day20 CUDA Preprocess + TensorRT Integration Summary

## Integration

完成 GPU pipeline：

```text
cv::Mat BGR
↓
CUDA Preprocess V1
↓
TensorRT Device Input
↓
TensorRT Inference
↓
D2H Output
```

CUDA Preprocess 输出直接作为 TensorRT input，不再进行 preprocess tensor D2H。

Correctness：**PASS**

---

## Copy-based Baseline

使用：

```text
Registered Host Memory
↓
Raw H2D
↓
CUDA Preprocess
↓
TensorRT
```

Benchmark：

```text
Raw H2D p50:      1.057 ms
Preprocess p50:   0.100 ms
TensorRT p50:     4.064 ms
D2H p50:          0.085 ms
Total p50:        5.307 ms
Total p95:        5.340 ms
```

主要额外开销已经从 CUDA Preprocess 转移到 `Raw H2D`。

---

## Zero-copy

使用 `cudaHostRegister` / mapped host memory，使 CUDA Preprocess 直接读取 host image，删除 Raw H2D copy。

Benchmark：

```text
Preprocess p50:   0.102 ms
TensorRT p50:     4.083 ms
D2H p50:          0.085 ms
Total p50:        4.270 ms
Total p95:        5.057 ms
```

相比 Copy-based pipeline：

```text
Total p50:
5.307 ms
→
4.270 ms

Improvement:
≈ 19.5%
```

Zero-copy 后 CUDA Preprocess 的典型 latency 基本没有退化，同时消除了约 `1.06 ms` 的 Raw H2D copy。

---

## Conclusion

Day20 完成了 CUDA Preprocess 与 TensorRT 的 device-side integration。

最终选择：

```text
Mapped Host Image
↓
CUDA Preprocess V1
↓
TensorRT
↓
D2H Output
```

CUDA Preprocess 已不再是主要 bottleneck。

相比继续优化约 `0.1 ms` 的 preprocess kernel，优化完整 memory/data path 带来了更明显的系统级收益。

**Selected Path: Zero-copy Input**