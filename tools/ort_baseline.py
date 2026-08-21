from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL_PATH="models/yolov8n.onnx"
OUTPUT_DIR=Path("results/ort/")

def run_batch(session,input_name,output_name,batch_size):
    rng=np.random.default_rng(seed=42)

    x=rng.random(
        (batch_size,3,640,640),
        dtype=np.float32,
    )
    
    print(f"\n===Batch{batch_size}===")
    print("input_shape:",x.shape)
    print("input_dtype:",x.dtype)
    print("intput_min",x.min())
    print("intput_max",x.max())
    print("input_mean",x.mean())
    print()

    outputs=session.run(
        [output_name],
        {
            input_name:x,
        },
    )

    y=outputs[0]

    print("onput_shape:",y.shape)
    print("onput_dtype:",y.dtype)
    print("ontput_min",y.min())
    print("ontput_max",y.max())
    print("onput_mean",y.mean())
    print()

    np.save(
        OUTPUT_DIR / f"input_b{batch_size}.npy",
        x,
    )
    np.save(
        OUTPUT_DIR / f"output_b{batch_size}.npy",
        y,
    )

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    session=ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    input_tensor=session.get_inputs()[0]
    output_tensor=session.get_outputs()[0]
        
    print("====MODEL I/O====")
    print()
    print("INPUT")
    print("name:",input_tensor.name)
    print("shape:",input_tensor.shape)
    print("type:",input_tensor.type)
    print()
    print("OUTPUT")
    print("name:",output_tensor.name)
    print("shape:",output_tensor.shape)
    print("type:",output_tensor.type)
    print()

    run_batch(
        session,
        input_tensor.name,
        output_tensor.name,
        batch_size=3,
    )

    # run_batch(
    #     session,
    #     input_tensor.name,
    #     output_tensor.name,
    #     batch_size=4,
    # )

if __name__=="__main__":
    main()