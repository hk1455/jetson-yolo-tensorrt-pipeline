import onnx

MODEL_PATH="models/yolov8n.onnx"

def print_value_info(value):
    tensor_type=value.type.tensor_type

    shape=[]

    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append("?")

    print(f"name: {value.name}")
    print(f"shape: {shape}")
    print()

def main():
    model=onnx.load(MODEL_PATH)

    print("___input___")
    for value in model.graph.input:
        print_value_info(value)

    print("__output__")
    for value in model.graph.output:
        print_value_info(value)

if __name__=="__main__":
    main()
