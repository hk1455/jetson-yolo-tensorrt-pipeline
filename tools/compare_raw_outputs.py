from pathlib import Path
import numpy as np


ORT_PATH="results/ort/bus_raw_output.npy"
CPP_PATH="results/cuda_preprocess/home_raw_output_V6.bin"

def main():
    ort=np.load(ORT_PATH)
    #print(ort.shape)

    cpp=np.fromfile(CPP_PATH, dtype=np.float32)
    cpp_reshaped=cpp.reshape(1,84,8400)
    # for i in [0,1,10,100,1000]:
    #     print(i,cpp[i])
    print(cpp_reshaped.shape)

    abs_error=np.abs(cpp_reshaped-ort)

    print(abs_error.max())
    print(abs_error.mean())
    print(np.median(abs_error))
    print(np.percentile(abs_error,99))

    flat_ort=ort.reshape(-1)
    flat_cpp=cpp_reshaped.reshape(-1)
    flat_error=abs_error.reshape(-1)

    indices=np.argsort(flat_error)[-10:][::-1]  #argsort返回的是索引（数组下标）

    for index in indices:
        idx3=np.unravel_index(index,ort.shape)
        print(
            "index= ",idx3,
            "trt= ",cpp_reshaped[idx3],
            "ort= ",ort[idx3],
            "abs_error= ",abs_error[idx3],
        )

    # ort = np.load("results/ort/output_b1.npy")
    # x = ort.reshape(-1)
    # for i in [0, 1, 10, 100, 1000, x.size // 2, x.size - 1]:
    #     print(i, x[i])


if __name__=="__main__":
    main()