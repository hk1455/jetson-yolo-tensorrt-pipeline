from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import json

IMAGE_PATH="assets/bus.jpg"
MODEL_PATH="models/yolov8n.onnx"

INPUT_SIZE=640

OUTPUT_DIR=Path("results/ort")

def letterbox(image,new_size=640):
    original_h,original_w=image.shape[:2]

    scale=min(
        new_size/original_h,
        new_size/original_w,
    )

    resized_h=int(round(original_h*scale))
    resized_w=int(round(original_w*scale))

    resized=cv2.resize(
        image,
        (resized_w,resized_h),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_w=new_size-resized_w
    pad_h=new_size-resized_h

    left=pad_w//2
    right=pad_w-left

    top=pad_h//2
    bottom=pad_h-top

    paded=cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114,114,114)
    )
    return paded,scale,left,top

def preprocess(image):

    image,scale,pad_x,pad_y=letterbox(
        image,
        INPUT_SIZE,
    )

    image=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image=image.astype(np.float32)

    image/=255.0

    image=np.transpose(
        image,
        (2,0,1),
    )

    image=np.expand_dims(
        image,
        axis=0,
    )

    image=np.ascontiguousarray(image)

    return image,scale,pad_x,pad_y

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image=cv2.imread(IMAGE_PATH)

    if image is None:
        raise RuntimeError(
            f"fail to read image:,{IMAGE_PATH}"
        )

    print("===Oringial Image===")
    print("shape:",image.shape)
    print("dtype:",image.dtype)

    letterboxed,_,_,_=letterbox(
        image,
        INPUT_SIZE,
    )

    cv2.imwrite(
        str(OUTPUT_DIR/"bus_letterbox.jpg"),
        letterboxed,
    )

    input_tensor,scale,pad_x,pad_y=preprocess(
        image
    )

    metadata={
        "scale":float(scale),
        "pad_x":int(pad_x),
        "pad_y":int(pad_y),
        "original_width":int(image.shape[1]),
        "original_height":int(image.shape[0]),
    }

    with open(OUTPUT_DIR/"bus_meta.json","w") as f:
        json.dump(metadata,f,indent=2)

    print("\n===Preprocess_tensor===")
    print("shape:", input_tensor.shape)
    print("dtype:", input_tensor.dtype) 
    print("min:", input_tensor.min())
    print("max:", input_tensor.max())
    print("mean:", input_tensor.mean())

    print("\n=== Letterbox Info ===")
    print("scale:", scale)
    print("pad_x:", pad_x)
    print("pad_y:", pad_y)

    np.save(
        OUTPUT_DIR/"bus_input.npy",
        input_tensor,
    )

    session=ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    input_name=session.get_inputs()[0].name
    output_name=session.get_outputs()[0].name

    outputs=session.run(
        [output_name],
        {
            input_name:input_tensor,
        },
    )

    raw_output=outputs[0]

    print("\n=== Raw Output ===")
    print("shape:", raw_output.shape)
    print("dtype:", raw_output.dtype)
    print("min:", raw_output.min())
    print("max:", raw_output.max())
    print("mean:", raw_output.mean())

    np.save(
        OUTPUT_DIR / "bus_raw_output.npy",
        raw_output,
    )

    print("\nSaved:")
    print(OUTPUT_DIR / "bus_input.npy")
    print(OUTPUT_DIR / "bus_raw_output.npy")
    print(OUTPUT_DIR / "bus_letterbox.jpg")

if __name__=="__main__":
    main()

