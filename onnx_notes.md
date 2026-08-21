# YOLOv8n ONNX Notes

## Input

Name: images

Shape:

N x 3 x height x width

Layout:

NCHW

Data type:

FP32

Dynamic dimensions:

Batch, Height, Width  (all dynamic)

## Output

Name: output0

Shape:

N x 84 x anchors

Data type:

FP32

> `anchors` 的数量会随输入的 height、width 自动变化。

## Export

Export method:  
`model.export(format="onnx", dynamic=True)`  
(or explicitly `dynamic={'height': True, 'width': True}`)

Image size used for tracing:  
640 × 640 (but any height/width can be used at inference)

Dynamic batch: Enabled  
Dynamic H/W: Enabled  

NMS: Disabled