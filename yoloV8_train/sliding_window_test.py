import numpy as np
import cv2
from time import time
from slice import slice_img_with_overlap
from PIL import Image
from sys import argv
import argparse

def add_boxes_v5(model,batch,all_boxes):
    imgs = [b[0] for b in batch]
    tcrops = [tc[1] for tc in batch]
    prediction = model(imgs)
    for p,tc in zip(prediction.xyxy,tcrops):
        res = p.cpu()
        for box in res:
            globalcoordbox = (int(box[0]) + tc[0],int(box[1]) + tc[1],int(box[2])+tc[0],int(box[3])+tc[1])
            all_boxes.append(globalcoordbox)

def add_classify_v8(model,frame,box):
    cut_img = frame[box[1]:box[3], box[0]:box[2]]

    results = model(cut_img)
    names = model.names
    for result in results:
        top_class = result.probs.top1
        top_confidence = result.probs.top1conf
        # print(f"{top_class} {top_confidence:.2f},xxxxxxxxxxxxxxxxxxx")
    return names[top_class],top_confidence.item()



def add_boxes_v8(model,batch,all_boxes):
    imgs = [b[0] for b in batch]
    tcrops = [tc[1] for tc in batch]
    prediction = model(imgs)
    for p,tc in zip(prediction,tcrops):
        for box in p.boxes:
            res = box.xyxy.cpu()[0].numpy().astype(np.int32)
            globalcoordbox = (res[0] + tc[0],res[1] + tc[1],res[2]+tc[0],res[3]+tc[1])
            all_boxes.append(globalcoordbox)

def visual_test(modelweights,videofile = "./videos/test.mp4",cropscount = 1, save = False, cls_model=None):
    # Inference
    from yolo_v8_test import get_model
    model = get_model(modelweights)
    model.predict(half=False,conf=0.0)
    if cls_model != None:
        model_class = get_model(cls_model)
        model_class.predict()

    #v = cv2.VideoCapture(videofile, cv2.CAP_GSTREAMER)
    v = cv2.VideoCapture(videofile)
    cv2.namedWindow("test",cv2.WINDOW_NORMAL)
    failCount = 0
    if save:
        frame_width = int(v.get(3))
        frame_height = int(v.get(4))
	   
        size = (frame_width, frame_height)
        # result = cv2.VideoWriter(f"{modelweights}_{videofile.split('/')[-1]}", 
        #                  cv2.VideoWriter_fourcc(*'MJPG'),
        #                  10, size)
        result = cv2.VideoWriter(f"{modelweights}_{videofile.split('/')[-1]}.mp4", cv2.VideoWriter_fourcc(*'mp4v'),30, size)
    
    while True:
        _, img = v.read()

        if failCount > 10:
            print("Exiting")
            break

        if not _ :
            failCount+=1
            continue

        t = time()
        crops  = slice_img_with_overlap(img.shape,cropscount,50)
        pimg = Image.fromarray(img)
        batch = []
        for ci,c in enumerate(crops):
            crop_coords  = (c[0],c[1],c[2],c[3])
            batch.append([np.array(pimg.crop(crop_coords)),c])

        all_boxes =  []

        add_boxes_v8(model,batch,all_boxes)
        if cls_model != None:
            all_boxes_with_class = []
            for i in all_boxes:
                conf = add_classify_v8(model_class,img, i)
                new_bbox = (i[0],i[1],i[2],i[3], conf)
                all_boxes_with_class.append(new_bbox)
        #results = model(batch)
        # print(all_boxes_with_class)
        print(f"Time:{(time() - t) * 1000} ms")
        if cls_model != None:

            for box in all_boxes_with_class:
                cv2.putText(img, f"""#{str(box[4][0])}""", (box[0] + 4, box[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.2, 100)
                cv2.putText(img, f"""{str(box[4][1])}""", (box[0] + 30, box[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.2, 100)
                cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0), 1)
        else:
            for box in all_boxes:
                cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0), 1)

        if save:    
            result.write(img)
        cv2.imshow("test",img)
        

        cv2.waitKey(1)
        # Results
        #results.print()
        #results.show()  # or .show()

def run_train(modelweights):
    from yolo_v8_test import run_training
    run_training(modelweights)

def save_model(modelweights):
    from yolo_v8_test import save_model_onnx
    save_model_onnx(modelweights)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default = "VISUAL_TEST" , help="run mode")
    parser.add_argument("--testvideo", default = "test.ts" , help="video file")
    parser.add_argument("--model", default = "yolov8n.yaml" , help="Model specification or weight file")
    parser.add_argument("--clsmodel", default = None , help="Model specification or weight file")
    parser.add_argument("--cropscount", default = 1 , type=int, help="number of crops per image")
    parser.add_argument("--save", default = True , type=bool, help="save the result to file where the model locate")
    args = parser.parse_args()

    if args.mode == "VISUAL_TEST":
        visual_test(args.model ,args.testvideo,args.cropscount, args.save, args.clsmodel)
    elif args.mode == "TRAIN":
        run_train(args.model)
    else :
        print('unsupported mode',args.mode)
    #visual_test()

