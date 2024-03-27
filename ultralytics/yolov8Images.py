
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

# datasetDir = "/home/yossi/Documents/database/hadar/videos/videos_drones/video_20231105175154/video_20231105175154/"
# datasetDir = "/home/yossi/Documents/database/hadar/videos/videos_drones/drones_birds/Dataset/test/"
datasetDir = "/home/yossi/Documents/database/hadar/videos/videos_drones/drone_data/birds/"

imgList        = getSorted(datasetDir=datasetDir, dirName="images", ext='.jpg')
labelDicList   = getSorted(datasetDir=datasetDir, dirName="labels", ext='.txt')

print(labelDicList[0:10])
ouStr=''
for idx, (imgDic, labelDic) in enumerate(zip(imgList, labelDicList)):
    
    if labelDic['data']:

        ##############################
        #               read
        ##############################
        print(f"{idx=}, {imgDic['name']=}, {labelDic['name']=}, {labelDic['data']=}")
        img = imgDic['image']
         # list of strings, in each the frone type and bbox, numbers are ratio from image  w,h
        labelDataStrList = labelDic['data'] # eg, '0 0.5998 0.83 0.00864 0.01275' <type, centerX,centerY,w,h>

        bboxList = []
        typeList = []
        for labelDataStr in labelDataStrList:
            labelData   = [float(strEle) for strEle in labelDataStr.split(' ')]
            print(f"{labelData=}")

            droneType   = int(labelData[0]) # 0/1 -> winged/quadcopter drone
            typeList.append(droneType)

            droneBox    = labelData[1:]
            h, w = img.shape[:2]

            centerX, centerY, boxW, boxH = int(droneBox[0]*512), int(droneBox[1]*512), int(droneBox[2]*10), int(droneBox[3]*10)
            xi, yi, xf, yf = centerX - boxW//2, centerY - boxH//2, centerX + boxW//2, centerY + boxH//2

            # rX, rY = 640/1024, 512/1024
            # xi, yi, boxW, boxH = int(droneBox[0]*512+60), int(droneBox[1]*512), int(droneBox[2]*10), int(droneBox[3]*5)
            # xf, yf = xi + boxW, yi + boxH
            bboxList.append([xi, yi, xf, yf])


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
            print(f"{boxIdx=}, {calssIdx=}, {className=}, {confidence=}, {box=}")

            if confidence >= 0.0:
                x0, y0, x1, y1 = [int(t.item()) for t in box]

                # prediction
                cv2.rectangle(img,pt1=(x0,y0), pt2=(x1,y1), color=(0,255,0),thickness=1)
                cv2.putText  (img, text=className, org=(x0,y0), 
                              fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0,255,0), thickness=1)

        # labels 
        for droneType, bbox in  zip(typeList, bboxList):
            color = (droneType*255,0,255*(1-droneType))
            cv2.rectangle(img,pt1=(bbox[0],bbox[1]), pt2=(bbox[2],bbox[3]), color=color,thickness=1)
            cv2.putText  (img, text=str(droneType), org=(bbox[0],bbox[1]),
                           fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=color, thickness=1)

        cv2.putText  (img, text=f"{idx=}", org=(100,100),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=color, thickness=3)
        cv2.imshow(f"result",img)

        if len(bboxList) > 1:
            ouStr += f"{idx=}, {imgDic['name']=}, {labelDic['name']}, {labelDic['data']=}\n"

        if cv2.waitKey(0) & 0xff == ord('q'): 
            break

cv2.destroyAllWindows()

print(f"note the following:\n{ouStr}")

# outputVideo.release()



    