from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL_PATH="results/ort/input_b3.npy"

def main():
    x=np.load(MODEL_PATH)
    assert x.dtype==np.float32
    x=np.ascontiguousarray(x)
    x.astype(np.float32).tofile("results/ort/input_b3.bin")


if __name__=="__main__":
    main()
    