
# Base - without tiling, simply imgsz set to 1024px, at result = model.predict(imgPath, imgsz=1024)

from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import torch
import os
from utils import getSorted

model = YOLO("yolov8s-p2.yaml") # https://github.com/ultralytics/ultralytics/issues/981
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

datasetDir = "/home/yossi/Documents/database/hadar/videos/videos_drones/video_20231105175154/video_20231105175154/"

imgList         = getSorted(datasetDir=datasetDir, dirName="images", ext='.jpg')
labelDataList   = getSorted(datasetDir=datasetDir, dirName="labels", ext='.txt')

for img, labelDataStr in zip(imgList, labelDataList):
    
    if labelDataStr:

        ##############################
        #               read
        ##############################
        labelData   = [float(strEle) for strEle in labelDataStr.split(' ')]
        print(f"{labelData=}")

        droneType   = labelData[0]
        droneBox    = labelData[1:]

        h, w = img.shape[:2]

        x0, y0, boxW, boxH = int(droneBox[0]*w), int(droneBox[1]*h), int(droneBox[2]*w), int(droneBox[3]*h)
        x1, y1 = x0 + boxW, y0 + boxH

        cv2.rectangle(img,pt1=(x0,y0), pt2=(x1,y1), color=(0,0,255),thickness=1)
        cv2.putText  (img, text=str(droneType), org=(x0,y0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,255), thickness=1)

        ##############################
        #               predict
        ##############################
        result = model.predict(img, imgsz=w)

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

                cv2.rectangle(img,pt1=(x0,y0), pt2=(x1,y1), color=(0,255,0),thickness=1)
                cv2.putText  (img, text=className, org=(x0,y0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0,255,0), thickness=1)

        cv2.imshow("result",img)

        if cv2.waitKey(0) & 0xff == ord('q'): 
            break

cv2.destroyAllWindows()

# outputVideo.release()



    