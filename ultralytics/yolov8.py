from ultralytics import YOLO
import cv2
import numpy as np
import datetime
# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8m.yaml")
model = YOLO(r"C:\Users\User\Documents\weights\yoloDronesM\realdrones_v1.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

# results = model("00132_jpg.rf.252ac892c1d9e97c7b899bab368e9358_with_white_background.jpg")  # predict on an image

# result = model.predict('drone1.jpg')


origImg = cv2.imread('droneSmall.jpg')
fx, fy = 0.5, 0.5
imgDownScaled = cv2.resize(origImg,None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

t,b = int(origImg.shape[0]*(1-fy) // 2), int(origImg.shape[0]*(1-fy) // 2)
l,r = int(origImg.shape[1]*(1-fx) // 2), int(origImg.shape[1]*(1-fx) // 2)
imgDsPadded = cv2.copyMakeBorder(imgDownScaled,t,l,r,b, borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
cv2.imwrite("imgDsPadded.jpg",imgDsPadded )

result = model.predict("imgDsPadded.jpg")

print(f"{result[0]=}")
print(f"{result[0].boxes=}")

print(f"{origImg.shape=}")
print(f"{imgDsPadded.shape=}")

x0, y0, x1, y1 = result[0].boxes.xyxy[0].numpy().astype(np.int32)
print('\n', x0, y0, x1, y1)

cv2.rectangle(imgDsPadded,pt1=(x0,y0), pt2=(x1,y1), color=(0,255,0),thickness=1)

curTime = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
filename = './runs/detect/savedHadar/detection_' + curTime + '.jpg'
cv2.imwrite(filename=filename, img=imgDsPadded)

cv2.imshow("result",imgDsPadded )
cv2.waitKey(0)
cv2.destroyAllWindows()