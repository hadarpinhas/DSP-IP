
# Base - without tiling, simply imgsz set to 1024px, at result = model.predict(imgPath, imgsz=1024)

from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import torch
import os
model = YOLO("yolov8s-p2.yaml") # https://github.com/ultralytics/ultralytics/issues/981
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

imgPath = "img.jpg"

cv2.waitKey(0)
cv2.destroyAllWindows()

videoPath = "/home/yossi/Documents/database/hadar/videos/videos_drones/vid1.264"

cap = cv2.VideoCapture(str(videoPath))

while(True):
    success, frame = cap.read()
    if not success:
        break

    result = model.predict(frame, imgsz=max(frame.shape))

    # print(f"*********\n{result=}")
    # print(f"*********\n{result[0].boxes=}")
    boxes = result[0].boxes
    calssNames = result[0].names
    calsses = boxes.cls
    confidences = boxes.conf
    boxes_xyxy = result[0].boxes.xyxy

    # print(f"{boxes_xyxy=}")

    for boxIdx, box in enumerate(boxes_xyxy):

        calssIdx = int(calsses[boxIdx].item())
        className = calssNames[calssIdx]
        confidence = confidences[boxIdx].item()
        print(f"{boxIdx=}, {calssIdx=}, {className=}, {confidence=}, {box=},")

        if confidence > 0.5:
            x0, y0, x1, y1 = [int(t.item()) for t in box]

            cv2.rectangle(frame,pt1=(x0,y0), pt2=(x1,y1), color=(0,255,0),thickness=1)
            cv2.putText  (frame, text=className, org=(x0,y0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0,255,0), thickness=1)

    cv2.imshow("result",frame)

    if cv2.waitKey(0) & 0xff == ord('q'): 
        break

cv2.destroyAllWindows()
cap.release()
# outputVideo.release()



    