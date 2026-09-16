from pathlib import Path
import numpy as np


CUDA_PATH="results/cuda_preprocess/home_raw_output_V6_fp16.bin"
CPP_PATH="results/cuda_preprocess/home_raw_output_V6.bin"

def main():
    cuda=np.fromfile(CUDA_PATH, dtype=np.float32)
    cuda_reshaped=cuda.reshape(84,8400)
    #print(ort.shape)

    cpp=np.fromfile(CPP_PATH, dtype=np.float32)
    cpp_reshaped=cpp.reshape(84,8400)
    # print(cpp.shape)  
    # for i in [0,1,10,100,1000]:
    #      print(i,cpp[i])
    print(cpp_reshaped.shape)

    abs_error=np.abs(cpp_reshaped-cuda_reshaped)

    print(abs_error.max())
    print(abs_error.mean())
    print(np.median(abs_error))
    print(np.percentile(abs_error,99))


    flat_error=abs_error.reshape(-1)

    indices=np.argsort(flat_error)[-10:][::-1]  #argsort返回的是索引（数组下标）

    for index in indices:
        idx3=np.unravel_index(index,cuda_reshaped.shape)
        print(
            "index= ",idx3,
            "fp32= ",cpp_reshaped[idx3],
            "fp16= ",cuda_reshaped[idx3],
            "abs_error= ",abs_error[idx3],
        )



if __name__=="__main__":
    main()