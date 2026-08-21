import json
from pathlib import Path

import cv2
from ultralytics import YOLO



IMAGE_PATH = "assets/home.jpeg"
MODEL_PATH = "yolov8n.pt"

OUTPUT_DIR = Path("results/pytorch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load image ourselves so we know the original input shape.
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    h, w, c = image.shape

    print("=== Input ===")
    print(f"image: {IMAGE_PATH}")
    print(f"shape: H={h}, W={w}, C={c}")
    print(f"dtype: {image.dtype}")

    # 2. Load pretrained YOLOv8n.
    model = YOLO(MODEL_PATH)

    # 3. Run inference.
    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        verbose=True,
        rect=False,
    )

    result = results[0]

    # 4. Convert final detections to a simple JSON format.
    detections = []

    boxes = result.boxes

    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().tolist()
        confidence = float(boxes.conf[i].cpu())
        class_id = int(boxes.cls[i].cpu())

        detection = {
            "class_id": class_id,
            "class_name": result.names[class_id],
            "confidence": confidence,
            "x1": float(xyxy[0]),
            "y1": float(xyxy[1]),
            "x2": float(xyxy[2]),
            "y2": float(xyxy[3]),
        }

        detections.append(detection)

    # 5. Save detection JSON.
    json_path = OUTPUT_DIR / "bus_result.json"

    with open(json_path, "w") as f:
        json.dump(
            {
                "image": IMAGE_PATH,
                "original_shape": {
                    "height": h,
                    "width": w,
                    "channels": c,
                },
                "model": MODEL_PATH,
                "imgsz": 640,
                "conf_threshold": 0.25,
                "num_detections": len(detections),
                "detections": detections,
            },
            f,
            indent=2,
        )

    # 6. Save visualization.
    plotted = result.plot()

    image_path = OUTPUT_DIR / "home_result.jpg"
    cv2.imwrite(str(image_path), plotted)

    # 7. Print summary.
    print("\n=== Detection Result ===")
    print(f"number of detections: {len(detections)}")

    for i, det in enumerate(detections):
        print(
            f"[{i}] "
            f"class={det['class_name']} "
            f"id={det['class_id']} "
            f"conf={det['confidence']:.4f} "
            f"box=({det['x1']:.1f}, {det['y1']:.1f}, "
            f"{det['x2']:.1f}, {det['y2']:.1f})"
        )

    print("\nSaved:")
    print(json_path)
    print(image_path)


if __name__ == "__main__":
    main()
