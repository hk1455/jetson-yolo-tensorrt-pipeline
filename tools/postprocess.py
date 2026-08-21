from pathlib import Path

import json
import cv2
import numpy as np
from collections import Counter

CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.45


def xywh_xyxy(box):
    cx=box[0]
    cy=box[1]
    w=box[2]
    h=box[3]
    x1=cx-w/2
    y1=cy-h/2
    x2=cx+w/2
    y2=cy+h/2
    return [x1,y1,x2,y2]

def compute_iou(box1,box2):
    inter_x1=max(box1[0],box2[0])
    inter_x2=min(box1[2],box2[2])
 
    inter_y2=min(box1[3],box2[3])
    inter_y1=max(box1[1],box2[1])

    inter_w=max(0,inter_x2-inter_x1)
    inter_h=max(0,inter_y2-inter_y1)

    intersection=inter_w*inter_h

    s1=(box1[2]-box1[0])*(box1[3]-box1[1])
    s2=(box2[2]-box2[0])*(box2[3]-box2[1])

    iou=intersection/(s1+s2-intersection)

    return iou

def decode(raw_output,conf_threshould):
    output=raw_output
    detections=[]
    output=np.transpose(
        output,
        (0,2,1),
    )
    pred=output[0]
    for n in range(pred.shape[0]):
        scores=pred[n,4:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]

        if confidence < conf_threshould:
            continue

        boxes=xywh_xyxy(pred[n,:4])

        detection={
                "x1":boxes[0],
                "y1":boxes[1],
                "x2":boxes[2],
                "y2":boxes[3],
                "confidence":confidence,
                "class_id":class_id,
                
        }
        detections.append(detection)
    return detections

def nms(detections,iou_threshold):
    i=0
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    while i < len(detections):
        max_detections=detections[i]
        max_class=max_detections["class_id"]
        box1=[max_detections["x1"],max_detections["y1"],max_detections["x2"],max_detections["y2"]]
        j=i+1
        while j < len(detections):
            det=detections[j]
            box2=[det["x1"],det["y1"],det["x2"],det["y2"]]
            class_id=det["class_id"]
            iou=compute_iou(box1,box2)
            if (iou>iou_threshold) and (class_id==max_class):
                del detections[j]
            else:
                j+=1
        i+=1

    return detections
        
def scale_boxes_to_original(detections,scale,pad_x,pad_y,original_width,original_height):
    for detection in  detections:
        detection["x1"]=(detection["x1"]-pad_x)/scale
        detection["y1"]=(detection["y1"]-pad_y)/scale
        detection["x2"]=(detection["x2"]-pad_x)/scale
        detection["y2"]=(detection["y2"]-pad_y)/scale

        detection["x1"]=max(0,min(detection["x1"],original_width))
        detection["x2"]=max(0,min(detection["x2"],original_width))
        detection["y1"]=max(0,min(detection["y1"],original_height))
        detection["y2"]=max(0,min(detection["y2"],original_height))

    return detections

def draw_detections(image, detections):
    CLASS_NAMES = {0: "person", 5: "bus", 11: "stop sign"}
    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        class_id = int(det["class_id"])
        confidence = float(det["confidence"])
        class_name = CLASS_NAMES.get(class_id, str(class_id))
        label = f"{class_name} {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

def main():
    output=np.load("results/ort/bus_raw_output.npy")
    image=cv2.imread("assets/bus.jpg")

    detections=decode(output,CONF_THRESHOLD)
    print(f"after decode:{len(detections)}")

    detections=nms(detections,IOU_THRESHOLD)

    print(f"after nms:{len(detections)}")

    with open("results/ort/bus_meta.json","r") as f:
        metadata=json.load(f)

    scale=metadata["scale"]
    pad_x=metadata["pad_x"]
    pad_y=metadata["pad_y"]
    original_width=metadata["original_width"]
    original_height=metadata["original_height"]

    detections=scale_boxes_to_original(detections,scale,pad_x,pad_y,original_width,original_height)

    result={
        "num_detection":len(detections),
        "detections":[],
    }
    for det in detections:
        result["detections"].append({
            "class_id":int(det["class_id"]),
            "confidence":float(det["confidence"]),
            "x1":float(det["x1"]),
            "y1":float(det["y1"]),
            "x2":float(det["x2"]),
            "y2":float(det["y2"]),
        })

    with open("results/ort/bus_detection.json","w") as f:
        json.dump( result,f,indent=2 )

    drawn=draw_detections(image,detections)

    cv2.imwrite(
        "results/ort/bus_detection.jpg",
        drawn,
    )

if __name__=="__main__":
    main()