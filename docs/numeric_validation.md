# TensorRT Numeric Validation

## 1. Goal

验证同一份输入 Tensor 在 **ONNX Runtime CPU FP32** 和 **TensorRT** 下得到的 Raw Output 是否数值一致。

验证流程：

```text
Same Input Tensor
      │
      ├── ONNX Runtime CPU → Reference Output
      │
      └── TensorRT         → TRT Output
                                │
                                ↓
                     Element-wise Comparison
```

本次只比较 Raw Output，不涉及 preprocess、decode、NMS 等后处理。

---

## 2. Test Setup

Model:

```text
yolov8n.onnx
```

Input:

```text
B1: [1, 3, 640, 640]
B4: [4, 3, 640, 640]
dtype: float32
```

Output:

```text
B1: [1, 84, 8400]
B4: [4, 84, 8400]
dtype: float32
```

ORT 和 TensorRT 使用完全相同的输入数据：

```text
input_b1.npy == input_b1.bin
input_b4.npy == input_b4.bin
```

其中 `.bin` 只是 `.npy` 中 float32 数据的 raw binary 版本。

---

## 3. Metrics

主要统计：

- Max Absolute Error
- Mean Absolute Error
- Median Absolute Error
- P99 Absolute Error
- Top-K Largest Differences

Absolute Error：

```text
|TensorRT - ORT|
```

---

## 4. Results

| Batch | TensorRT Mode | Max Abs | Mean Abs | Median Abs | P99 Abs |
|---|---|---:|---:|---:|---:|
| B1 | FP32 / TF32 | 0.80181885 | 0.0017354608 | 2.962463e-08 | 0.053649902 |
| B1 | FP32 no-TF32 | 0.0012512207 | 1.3031694e-06 | 2.7747422e-08 | 4.5776367e-05 |
| B4 | FP32 / TF32 | 0.8987427 | 0.0016607014 | 2.9455162e-08 | 0.050949097 |
| B4 | FP32 no-TF32 | 0.001159668 | 1.3115998e-06 | 2.7600887e-08 | 4.5776367e-05 |

---

## 5. Observations

### Default FP32 / TF32

TensorRT 默认允许使用 TF32。

B1 和 B4 的：

```text
Mean Abs ≈ 1.7e-3
P99 Abs  ≈ 5e-2
Max Abs  < 0.9
```

整体输出与 ORT reference 接近，但部分 bbox 相关位置存在较明显的数值偏差。

### FP32 no-TF32

关闭 TF32 后：

```text
Mean Abs ≈ 1.3e-6
P99 Abs  ≈ 4.6e-5
Max Abs  ≈ 1e-3
```

与 ORT CPU FP32 的结果非常接近。

说明默认模式下较明显的误差主要来自 TF32 计算策略，而不是 TensorRT Runtime 执行错误。

---

## 6. Dynamic Batch Validation

B1 和 B4 的误差分布基本一致。

从：

```text
B1 → B4
```

没有出现明显的数值恶化，说明当前：

```text
Dynamic Shape
Buffer Size Calculation
CUDA Memory Copy
setTensorAddress
enqueueV3
Raw Output
```

在 B1 / B4 下工作正常。

---

## 7. Conclusion

Day 11 Numeric Validation 通过。

```text
ORT CPU FP32
      ≈
TensorRT FP32 / TF32

ORT CPU FP32
      ≈≈
TensorRT FP32 no-TF32
```

验证结果表明：

1. TensorRT Engine 的 Raw Output 与 ORT reference 数值一致。
2. TensorRT 默认 TF32 会引入一定浮点误差。
3. 关闭 TF32 后，TensorRT 与 ORT CPU FP32 几乎一致。
4. B1 和 B4 均通过验证，Dynamic Batch Runtime 工作正常。

因此后续如果图片推理结果出现问题，可以优先检查：

```text
Preprocess
Letterbox
Decode
Reverse Letterbox
NMS
```

而不是首先怀疑 TensorRT Engine / Runtime。

## FP16 Validation
| Batch | Engine | Max Abs | Mean Abs | Median Abs | P99 Abs |
|---|---|---:|---:|---:|---:|
| B1 | FP16 | 3.5651245 | 0.004404223 | 5.9604645e-08 | 0.13253784 |