from ultralytics import YOLO
import numpy as np
import cv2

#from ultralytics.yolo import data

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
def get_model_v5():
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m6', pretrained=True)
    model.conf = 0.1

def get_model(modelweights = 'yolov8n.pt',pretrained=True):
    # load a pretrained model (recommended for training)
    model = YOLO(modelweights)
    model.predict(half=True)
    model.conf = 0.05
    return model

def save_model_onnx(modelweights = 'yolov8n.pt',batch = 1, pretrained=True, savepath=None):
    # load a pretrained model (recommended for training)
    model = YOLO(modelweights)
    print(type(model))
    model.export(format='onnx',path=savepath,dynamic=True ,half=True)

# Use the model
#model.train(data="coco128.yaml", epochs=3)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
def run_test(modelweights):
    model = get_model(modelweights)
    results = model("bus.jpg")  # predict on an image
    img = results[0].orig_img
    for b in results[0].boxes:
        xyxy = b.xyxy.cpu()[0].numpy().astype(np.int32)
        img = cv2.rectangle(img,(xyxy[0],xyxy[1]),(xyxy[2],xyxy[3]), (255,0,0),5)

    cv2.imshow('win',img)
    cv2.waitKey(0)

def run_training(modelweights):
    model = get_model(modelweights,False)
    model.train(data="data.yaml",shear=.001,perspective = 0.001,hsv_s=1, batch=4)
