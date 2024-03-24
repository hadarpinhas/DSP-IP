# rather than using base which does use tiling for better input resolution

from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import os
import torch
from utils import divide_image_into_tiles

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

model = YOLO("yolov8s-p2.yaml") # https://github.com/ultralytics/ultralytics/issues/981
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# model = YOLO("yolov8m.yaml")
# model = YOLO(r"C:\Users\User\Documents\weights\yoloDronesM\realdrones_v1.pt")  # load a pretrained model (recommended for training)
# model = YOLO(r"/home/yossi/Documents/weights/yoloDronesM/realdrones_v1.pt")  # load a pretrained model (recommended for training)
#                

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

# results = model("00132_jpg.rf.252ac892c1d9e97c7b899bab368e9358_with_white_background.jpg")  # predict on an image

# result = model.predict('drone1.jpg')

targetImgPath = 'droneSmall.jpg'
origImg = cv2.imread(targetImgPath)

# fx, fy = 0.7, 0.7
# imgDownScaled = cv2.resize(origImg,None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
# t,b = int(origImg.shape[0]*(1-fy) // 2), int(origImg.shape[0]*(1-fy) // 2)
# l,r = int(origImg.shape[1]*(1-fx) // 2), int(origImg.shape[1]*(1-fx) // 2)
# imgDsPadded = cv2.copyMakeBorder(imgDownScaled,t,l,r,b, borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
# imgDsPaddedPath = "imgDsPadded"
# cv2.imwrite(imgDsPaddedPath,imgDsPadded)

tilesList = divide_image_into_tiles(image_path=targetImgPath, tile_width=600, tile_height=600)

os.makedirs("tiles",exist_ok=True)

for tileIdx, tile in enumerate(tilesList):

    tilePath =  "tiles/imgDsPadded_" + str(tileIdx) + ".jpg"
    cv2.imwrite(tilePath,tile )
    result = model.predict(tilePath)

    print(f"{result[0]=}")
    print(f"{result[0].boxes=}")

    print(f"{origImg.shape=}")
    print(f"{tile.shape=}")


    if torch.cuda.is_available():
        x0, y0, x1, y1 = result[0].boxes.xyxy[0].cpu().numpy().astype(np.int32)
    else:
        x0, y0, x1, y1 = result[0].boxes.xyxy[0].numpy().astype(np.int32) # for cpu

    print('\n', x0, y0, x1, y1)

    cv2.rectangle(tile,pt1=(x0,y0), pt2=(x1,y1), color=(0,255,0),thickness=1)

    curTime = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    savePath = './runs/detect/savedHadar/'
    os.makedirs(savePath,exist_ok=True)

    filename = savePath + tilePath + '_' + curTime + '.jpg'
    cv2.imwrite(filename=filename, img=tile)

    cv2.imshow("tile" + str(tileIdx) ,tile)
    if cv2.waitKey(0) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()