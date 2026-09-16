# Day24 Summary — Multi-Stream Pipeline

## Setup

- Platform: Jetson AGX Orin
- Model: YOLOv8n
- TensorRT: FP16
- Batch: 1
- Input: 640×640
- FrameSlot: 2
- `IExecutionContext`: 1
- CUDA Streams: 2

## Pipeline

```text
Preprocess Stream:
Preprocess(N+1)
        |
        | preprocess_done
        v

Inference Stream:
TRT(N) -> D2H(N) -> output_ready
```

核心目标：

```text
TRT(N) || Preprocess(N+1)
```

`cudaStreamWaitEvent()` 用于跨 Stream GPU dependency；CPU 只在读取 `host_output` 前等待 `output_ready`。

## Current Host Timing

With `sudo jetson_clocks`:

| Stage | Mean |
|---|---:|
| TensorRT host submission (`inferfram`) | 1.345 ms |
| CPU wait for `output_ready` | 1.471 ms |
| CPU postprocess total | 2.872 ms |
| Decode / process | 2.835 ms |
| NMS | 0.035 ms |
| Reverse letterbox | ~0.0004 ms |

当前 CPU postprocess 的主要瓶颈是 decode / `process()`。

## GPU Stage Reference

| Stage | p50 |
|---|---:|
| CUDA preprocess | 0.102 ms |
| TensorRT | 2.284 ms |
| D2H | 0.087 ms |

CUDA preprocess 已经非常短，因此 `TRT(N) || Preprocess(N+1)` 可隐藏的时间本身很有限。

## Throughput

With `jetson_clocks` enabled:

| Version | Total / 1000 frames | Throughput |
|---|---:|---:|
| Day23 — Double Buffer, Single Stream | 5520.80 ms | 181.133 FPS |
| Day24 — Double Buffer, Two Streams | 5601.89 ms | 178.511 FPS |

Day24 throughput change:

```text
≈ -1.45%
```

## Conclusion

Day24 successfully established:

- Double-buffered FrameSlot reuse
- Separate preprocess / inference streams
- CUDA Event cross-stream dependency
- `TRT(N)` and `Preprocess(N+1)` overlap

However, throughput did not improve. Preprocess is only about `0.1 ms`, so the available overlap benefit is very small and can be offset by event / multi-stream scheduling overhead and GPU resource contention.

The current larger host-side bottleneck is CPU decode/postprocess (~2.8 ms), which still blocks submission of the next TensorRT inference.

## Next

Day25 should focus on separating GPU submission from CPU postprocess so that:

```text
TRT(N+1) || CPU Postprocess(N)
```

can become possible.