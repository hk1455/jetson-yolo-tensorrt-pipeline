# Day16 - Runtime Memory Engineering

Date:  
2026-08-25

## 1. Goal

Day16 focuses on TensorRT runtime memory management and host/device data transfer.

Main experiment:

Pageable Host Memory  
vs  
Pinned Host Memory

Main metrics:

- H2D latency
- D2H latency
- Host Infer latency
- Core E2E latency

The runtime follows the principle:

Allocate Once  
→ Reuse During Inference  
→ Release At Shutdown

---

## 2. Benchmark Environment

Platform:  
Jetson AGX Orin

Power Mode:  
MAXN

jetson_clocks:  
ON

CPU Clock:  
2.2016 GHz locked

GPU Clock:  
1.3005 GHz locked

EMC:  
3.199 GHz

Engine:  
TensorRT FP32

Batch:  
1

Input:  
1x3x640x640

Runs:  
200

Pinned benchmark warmup:  
100

---

## 3. Pageable Baseline

Day15 pageable-memory baseline:

Preprocess:
2.41749 ms

Host Infer:
5.44037 ms

Postprocess:
3.63105 ms

Core E2E:
11.4889 ms

H2D:
0.54423 ms

TensorRT:
4.41896 ms

D2H:
0.428815 ms

Serial-equivalent FPS:
~87.0 FPS

---

## 4. Pinned Memory Implementation

Pinned host buffers are allocated using:

cudaMallocHost()

Pinned input and output buffers are allocated once before the benchmark loop and reused for all inference runs.

Current input path:

CPU Preprocess  
→ std::vector<float>  
→ pageable-to-pinned staging memcpy  
→ pinned_input  
→ H2D  
→ TensorRT

Current output path:

TensorRT  
→ D2H  
→ pinned_output  
→ CPU Postprocess

The output is consumed directly from pinned memory without copying back into a std::vector.

---

## 5. Pinned Benchmark Results

### Mean

Preprocess:
2.49607 ms

Staging Memory Copy:
0.463496 ms

Host Infer:
4.46686 ms

Postprocess:
2.89505 ms

Core E2E:
10.3215 ms

NMS:
0.0482182 ms

PROCESS:
2.84627 ms

Reverse Letterbox:
0.000570575 ms

### GPU Runtime

H2D:
0.174107 ms

TensorRT:
4.16786 ms

D2H:
0.0834982 ms

### p50

Preprocess:
2.47688 ms

Staging Memory Copy:
0.458632 ms

Host Infer:
4.39645 ms

Postprocess:
2.89538 ms

Core E2E:
10.2834 ms

H2D:
0.173728 ms

TensorRT:
4.09786 ms

D2H:
0.083696 ms

### p95

Preprocess:
2.56390 ms

Staging Memory Copy:
0.495812 ms

Host Infer:
4.59527 ms

Postprocess:
2.91833 ms

Core E2E:
10.5141 ms

H2D:
0.178915 ms

TensorRT:
4.29139 ms

D2H:
0.087648 ms

### Range

Core E2E:

min:
10.1347 ms

max:
11.2474 ms

TensorRT:

min:
4.07885 ms

max:
4.34067 ms

---

## 6. Pageable vs Pinned Comparison

| Stage | Pageable | Pinned |
|---|---:|---:|
| H2D | 0.544 ms | 0.174 ms |
| D2H | 0.429 ms | 0.083 ms |
| Host Infer | 5.440 ms | 4.467 ms |
| Core E2E | 11.489 ms | 10.322 ms |
| Serial-equivalent FPS | ~87.0 FPS | ~96.9 FPS |

H2D latency reduction:

~68%

D2H latency reduction:

~80%

Host Infer reduction:

~18%

Observed Core E2E reduction:

~10%

---

## 7. Staging Memory Analysis

Pinned input currently requires an additional host-side copy:

std::vector<float>  
→ pinned_input

Mean staging latency:

0.4635 ms

Therefore the current pinned input-side memory path is approximately:

Staging + H2D

= 0.4635 + 0.1741

= 0.6376 ms

The original pageable H2D latency was:

0.5442 ms

Therefore, with the current CPU preprocess architecture, the additional staging copy offsets the H2D improvement on the input side.

Pinned H2D itself is significantly faster, but the benefit is reduced because preprocess does not write directly into pinned memory.

---

## 8. Output-Side Result

The output path does not require an additional staging copy.

Pageable D2H:

0.4288 ms

Pinned D2H:

0.0835 ms

This is a clear improvement of approximately 80%.

The CPU postprocess directly consumes pinned_output.

Therefore, the output-side pinned-memory optimization provides a direct runtime benefit.

---

## 9. Total Memory-Path Cost

Pageable host/device transfer cost:

H2D + D2H

= 0.5442 + 0.4288

= 0.9730 ms

Current pinned-memory path:

Staging + H2D + D2H

= 0.4635 + 0.1741 + 0.0835

= 0.7211 ms

Total observed memory-path reduction:

~25.9%

Therefore, even after including the additional pageable-to-pinned staging copy, the overall host/device memory path is faster in the current implementation.

---

## 10. Runtime Memory Lifecycle

The runtime memory-management strategy is:

Program Startup

→ Allocate device input buffer once  
→ Allocate device output buffer once  
→ Allocate pinned host input buffer once  
→ Allocate pinned host output buffer once  
→ Create TensorRT execution context once

Benchmark / Inference Loop

→ Reuse all existing buffers  
→ No cudaMalloc/cudaFree inside infer()

Program Shutdown

→ cudaFree device buffers  
→ cudaFreeHost pinned buffers

This follows:

Allocate Once, Reuse Forever

where "Forever" means for the lifetime of the runtime/engine object.

---

## 11. Observations

1. Pinned host memory significantly reduces raw CUDA transfer latency.

2. H2D decreased from approximately 0.544 ms to 0.174 ms.

3. D2H decreased from approximately 0.429 ms to 0.083 ms.

4. The current input path requires approximately 0.463 ms of pageable-to-pinned staging copy.

5. Because of this staging copy, the input-side H2D improvement is mostly offset.

6. The output-side pinned-memory optimization provides a clear benefit because no additional staging copy is required.

7. Including staging, the total measured host/device memory-path cost decreased from approximately 0.973 ms to 0.721 ms.

8. Pinned Core E2E reached approximately 10.32 ms, corresponding to approximately 96.9 serial-equivalent FPS.

9. With Warmup = 100, TensorRT latency was stable:

   p50 = 4.098 ms  
   p95 = 4.291 ms  
   max = 4.341 ms

10. NMS and reverse-letterbox remain negligible performance costs.

---

## 12. Important Comparison Limitation

The Day16 implementation also changed the postprocess interface from std::vector<float> to const float*.

In addition, the final pinned benchmark used Warmup = 100, while the previous Day15 baseline used a different benchmark configuration.

Therefore:

The observed Core E2E improvement from 11.489 ms to 10.322 ms should not be attributed entirely to pinned memory.

The most directly attributable pinned-memory results are the H2D and D2H improvements.

A fully controlled pageable-vs-pinned A/B benchmark using identical interfaces and warmup conditions would be required to measure the exact causal E2E improvement from pinned memory alone.

For the purpose of Day16, the current experiment is sufficient to demonstrate the memory-transfer behavior and identify the staging-copy limitation.

---

## 13. Conclusion

Day16 successfully established a runtime memory-management model based on reusable buffers.

Pinned host memory provides a significant improvement in CUDA transfer latency:

H2D:
~68% lower

D2H:
~80% lower

However, the current CPU preprocess still writes into pageable std::vector memory first, requiring an additional staging memcpy before H2D.

This staging copy limits the input-side benefit of pinned memory.

The output side benefits directly from pinned memory because postprocess consumes pinned_output without another host-side copy.

Current important insight:

Pinned Memory itself is effective.

The remaining limitation is the CPU-side data path before pinned_input.

---

## 14. Next Hypothesis

A future implementation could allow preprocess to write directly into the final host input buffer:

CPU Preprocess  
→ pinned_input  
→ H2D

This would eliminate the current pageable-to-pinned staging copy of approximately 0.46 ms.

However, if the next stage moves preprocessing to CUDA, optimizing the current CPU preprocess output path may not be the highest-priority task.

The next optimization should continue to be driven by benchmark data rather than assumed performance gains.