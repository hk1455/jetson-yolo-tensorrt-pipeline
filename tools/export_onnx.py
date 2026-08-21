from ultralytics import YOLO

MODEL_PATH="models/yolov8n.pt"

def main():
    model=YOLO(MODEL_PATH)

    model.export(
        format="onnx",
        imgsz=640,
        dynamic=True,
        simplify=True,
        nms=False,
    )


if __name__=="__main__":
    main()



