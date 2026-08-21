# YOLOv8n TensorRT C++ Deployment

在 NVIDIA Jetson AGX Orin 上完成 YOLOv8n 的 ONNX 导出、TensorRT engine 构建、C++ 推理、前后处理、性能测试与数值校验。

推理流程：

~~~text
image -> letterbox/RGB/FP32 NCHW -> TensorRT -> decode -> class-aware NMS -> annotated image
~~~

> 重要：build/ 下的可执行文件是 aarch64 Linux ELF，models/ 下的 .engine 也与 GPU、TensorRT 和 CUDA 环境绑定。复制到 Windows 或另一台 Jetson 后，应在目标 Jetson 上重新编译并重新生成 engine。

## Development environment

已验证环境：

| Component | Version |
|---|---|
| Platform | NVIDIA Jetson AGX Orin, aarch64 |
| Jetson Linux / L4T | R36.4.7 |
| CUDA / cuDNN | 12.6 / 9.3.0 |
| TensorRT / trtexec | 10.3.0 |
| GCC / CMake | 11.4.0 / 3.27.9 |
| OpenCV | 4.8.0 |
| Python / ONNX / ONNX Runtime | 3.10.12 / 1.20.0 / 1.23.2 |

Python 端只用于导出和基准校验；实际 GPU 推理由 TensorRT C++ 完成。

## Build instructions

### 1. 准备依赖

推荐使用装有 JetPack 的 Jetson。确认 CUDA、TensorRT 开发包、CMake、G++ 和 OpenCV 开发包可用：

~~~bash
nvcc --version
/usr/src/tensorrt/bin/trtexec --version
cmake --version
pkg-config --modversion opencv4
~~~

仓库已包含 models/yolov8n.onnx，因此只运行 C++ 推理时不需要安装 Python 依赖。

### 2. 在目标设备重建 TensorRT engine

下面生成 yolo_detect 默认读取的 FP32、动态 batch、关闭 TF32 的 engine。关闭 TF32 可获得最接近 ONNX Runtime FP32 的数值结果。

~~~bash
cd /path/to/yolo-tensorrt-deploy

/usr/src/tensorrt/bin/trtexec \
  --onnx=models/yolov8n.onnx \
  --saveEngine=models/yolov8n_cpp_fp32.engine \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:8x3x640x640 \
  --noTF32 \
  --skipInference
~~~

engine 不保证跨设备、跨 TensorRT 版本或跨 CUDA 版本可用。

### 3. 编译

使用新的构建目录，避免复制过来的 build/CMakeCache.txt 引用旧机器路径：

~~~bash
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release
cmake --build build-local -j"$(nproc)"
~~~

主要程序：

| Program | Purpose |
|---|---|
| build_engine | 使用 C++ API 构建 engine |
| inspect_engine | 查看 engine 输入输出 |
| test_preprocess | 单独测试 C++ 前处理 |
| runtime_engine | 读取输入 bin 并输出 raw output bin |
| postprocess | 对 raw output 解码、NMS、画框 |
| yolo_detect | 完整单图推理 |

### 4. 最简单的运行方法

从项目根目录运行：

~~~bash
./build-local/yolo_detect
~~~

默认输入为 assets/home.jpeg，结果写入 results/cpp/home_detection.jpeg；同时保存前处理输入和 raw output。当前程序通过源码常量指定路径，如需换图，修改 apps/yolo_detect.cpp 顶部的 IMAGE_PATH 和 OUT_PUT 后重新编译。

### 5. 可选：重新导出 ONNX

仅在需要从 .pt 重新导出时执行：

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ultralytics onnx onnxruntime opencv-python numpy
python tools/export_onnx.py
~~~

## Model assumptions

| Item | Assumption |
|---|---|
| Model | Ultralytics YOLOv8n detection model，COCO 80 classes |
| ONNX input | images，FP32，NCHW |
| C++ input size | N x 3 x 640 x 640 |
| Preprocess | BGR -> RGB；等比例 letterbox；padding=114；除以 255；HWC -> CHW |
| Engine batch profile | min=1，opt=4，max=8 |
| Integrated app | yolo_detect 当前固定 batch=1 |
| ONNX output | output0，N x 84 x 8400（输入为 640 x 640 时） |
| Output channels | 4 个 xywh + 80 个类别分数；没有单独 objectness |
| NMS | ONNX 内不含 NMS；C++ 使用 class-aware NMS |
| Default thresholds | yolo_detect: confidence=0.20，IoU=0.70 |

ONNX 本身导出了动态 batch、高和宽，但当前 C++ runtime 将高宽固定为 640，后处理也将候选数固定为 8400。因此现阶段真正支持的是 640 x 640；若要支持其他分辨率，需要同时修改 runtime buffer 计算和后处理候选数。

apps/postprocess_raw.cpp 与 Python correctness baseline 使用 confidence=0.25、IoU=0.45，和 yolo_detect 的默认阈值不同。比较最终检测结果时必须使用相同阈值。

## Output format

### Tensor

| File / tensor | Type and layout | Batch 1 size |
|---|---|---:|
| Preprocessed input .bin | little-endian FP32，连续 NCHW，[1,3,640,640] | 4,915,200 bytes |
| TensorRT raw output .bin | little-endian FP32，连续 [N,84,8400] | 2,822,400 bytes |
| ONNX Runtime .npy | NumPy array，shape 信息保存在文件头 | varies |

Raw output 的内存顺序为 [batch][channel][candidate]。每个 candidate 的前 4 个通道是中心点坐标和宽高，后 80 个通道是类别分数。

### Final detection

后处理得到的 detection 字段为：

~~~json
{
  "class_id": 0,
  "confidence": 0.8902,
  "x1": 670.45,
  "y1": 380.56,
  "x2": 809.92,
  "y2": 879.65
}
~~~

坐标为映射回原图后的 xyxy 像素坐标。当前 C++ 完整程序输出标注图和二进制 raw output，不写检测 JSON；results/ort/bus_detection.json 是参考 JSON 格式。

默认完整推理输出：

~~~text
results/cpp/home_input.bin
results/cpp/home_raw_output.bin
results/cpp/home_detection.jpeg
~~~

## Benchmark table

测试环境为 Jetson AGX Orin、TensorRT 10.3、输入 640 x 640；数据来自 results/trtexec/ 中保存的日志，包含 H2D、GPU 和 D2H。表中的 FP32 使用 TensorRT 默认开启的 TF32。功耗模式和 GPU 锁频状态未记录，因此下表用于同环境相对比较，不代表所有 Orin 的固定性能。

| Precision | Batch | Mean host latency | Mean GPU compute | P95 host latency | Throughput |
|---|---:|---:|---:|---:|---:|
| FP32 (TF32 on) | 1 | 4.268 ms/image | 3.843 ms | 4.584 ms | 259.5 images/s |
| FP16 mixed | 1 | 2.476 ms/image | 2.153 ms | 2.495 ms | 463.7 images/s |
| FP32 dynamic (TF32 on) | 4 | 16.730 ms/batch | 15.223 ms/batch | 22.901 ms/batch | 65.32 batch/s = 261.3 images/s |

在 batch=1 的这组记录中，FP16 相对 FP32 的吞吐约为 1.79 倍，trtexec 平均 host latency 约降低 42%。

复现单 batch benchmark：

~~~bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=models/yolov8n_fp32.engine \
  --shapes=images:1x3x640x640
~~~

## Correctness results

验证基准为 ONNX Runtime CPU FP32，TensorRT 使用完全相同的 FP32 输入。误差定义为 abs(TensorRT - ONNX Runtime)。

| Batch | TensorRT mode | Max abs | Mean abs | P99 abs |
|---:|---|---:|---:|---:|
| 1 | FP32 / TF32 | 8.018e-1 | 1.735e-3 | 5.365e-2 |
| 1 | FP32 no-TF32 | 1.251e-3 | 1.303e-6 | 4.578e-5 |
| 1 | FP16 | 3.565 | 4.404e-3 | 1.325e-1 |
| 4 | FP32 / TF32 | 8.987e-1 | 1.661e-3 | 5.095e-2 |
| 4 | FP32 no-TF32 | 1.160e-3 | 1.312e-6 | 4.578e-5 |
| 4 | FP16 | 3.565 | 4.445e-3 | 1.336e-1 |

结论：

- C++ 与 Python 对 bus.jpg 的前处理最大绝对误差为 5.96e-8。
- 关闭 TF32 后，B1/B4 raw output 与 ORT CPU FP32 最接近，动态 batch 没有出现额外误差放大。
- bus.jpg 在 confidence=0.25、IoU=0.45 下，ORT 与 TensorRT 均得到 5 个检测，类别序列一致；最大置信度差为 2.84e-4，最大框坐标差为 0.125 pixel。
- FP16 的 raw output 浮点误差更大，但这是精度模式差异；是否满足业务要求应在目标数据集上评估。
- 当前结果证明实现链路的数值一致性，不等同于 COCO mAP 精度报告；本项目尚未记录数据集级 mAP。

更完整的 FP32 数值校验记录见 docs/numeric_validation.md。

快速复核前处理和单图 raw output：

~~~bash
python tools/image_preprocess.py
./build-local/test_preprocess
python tools/compare_preprocess.py

./build-local/runtime_engine
python tools/compare_raw_outputs.py
~~~

所有命令都应从项目根目录执行，因为当前程序使用相对路径。

