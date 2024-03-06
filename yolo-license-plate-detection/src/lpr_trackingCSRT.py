# https://github.com/computervisioneng/yolo-license-plate-detection/tree/master?tab=readme-ov-file
# weights: https://drive.google.com/file/d/1vXjIoRWY0aIpYfhj3TnPUGdmJoHnWaOc/edit

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 512       # Width of network's input image
inpHeight = 512      # Height of network's input image

# Load classes
classesFile = "../model/classes.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

inputVidPath = r'/home/yossi/Documents/database/hadar/videos/parkingLot/inupt2.mp4'

manual_select_roi = False

# Load the network
modelConfiguration = "../model/config/darknet-yolov3.cfg"
# modelWeights = r"C:\Users\User\Documents\weights\lpr_yolo3\model.weights"
modelWeights = "/home/yossi/Documents/weights/lpr_yolo3/model.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Initialize multi tracker
multiTracker = cv.legacy.MultiTracker_create()

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area != 0 else 0

def updateTrackersAndDetections(frame, outs, multiTracker):
    frameHeight, frameWidth = frame.shape[:2]
    classIds, confidences, boxes = [], [], []

    # Process detections
    for out in outs: # there are 3 outs: (507,6), (2028,6), (8112,6)
        for detection in out: # detection shape = (8112, 6) = (8112, [center_x, center_y, width, height, Objectness Score, Class Scores])
            scores = detection[5:]
            classId = np.argmax(scores) # take the index of the maximal score
            confidence = scores[classId] # confidence is the the highest score
            if confidence > confThreshold:
                center_x, center_y, width, height = (detection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])).astype(int)
                left, top = int(center_x - width / 2), int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append((left, top, width, height))  # Use tuple for consistency

    # Apply NMS
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    nms_boxes = [boxes[i] for i in indices] # Filtered new detections

    # Update tracker boxes
    success, tracked_boxes = multiTracker.update(frame)

    updated_trackers = []
    # Prepare to recreate MultiTracker with updated information
    for tbox in tracked_boxes:
        left, top, width, height = [int(v) for v in tbox]
        # Check if the box is within the frame
        if left >= 0 and top >= 0 and (left + width) <= frameWidth and (top + height) <= frameHeight:        
            updated_trackers.append(tbox) # Initialize with existing trackers

    # Iterate through nms_boxes to check for significant overlap
    for nms_box in nms_boxes:
        is_tracked = False
        for i, tbox in enumerate(updated_trackers):
            if iou(nms_box, tbox) > 0.2:
                # Significant overlap found, update this tracker's box with nms_box
                updated_trackers[i] = nms_box
                is_tracked = True
                break
        if not is_tracked: # new bbox tracker
            updated_trackers.append(nms_box)
            left, top, width, height = [int(v) for v in nms_box]
            cv.rectangle(frame, (left, top), (left + width, top + height), (255,0,0), 2, 1)

    # Recreate MultiTracker and add all trackers (updated and new)
    multiTracker = cv.legacy.MultiTracker_create()
    for box in updated_trackers:
        tracker = cv.legacy.TrackerCSRT_create()  # Recreate tracker for updated boxes
        multiTracker.add(tracker, frame, box)

    # Optionally, draw all boxes
    for box in updated_trackers:
        left, top, width, height = [int(v) for v in box]
        cv.rectangle(frame, (left, top), (left + width, top + height), (0,255,0), 2, 1)

    return multiTracker

#------------------------------------------------------------------------------------------------------

def doSobel(src):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    src = cv.GaussianBlur(src, (3, 3), 0)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x) # =abs(scaledImg).astype(np.uint8). default: scales with alpah=1 beta=0 scaledImg=Img*alpha+beta. 
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)

    return grad

#------------------------------------------------------------------------------------------------------
  
def match(baseImg, templateImg):

    gradTemplate = doSobel(templateImg)
    gradBase     = doSobel(baseImg)

    # res = Correlation matrix
    res = cv.matchTemplate(gradBase, gradTemplate, cv.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    return max_loc

margin = 20

# Main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection and Tracking with YOLO and OpenCV')
    parser.add_argument('--video', help='Path to video file.', default=inputVidPath)
    args = parser.parse_args()

    if args.video:
        cap = cv.VideoCapture(args.video)
        outputFile = args.video[:-4]+'_yolo_out_py.mp4'
    else:
        cap = cv.VideoCapture(0)

    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc(*'mp4v'), 30, (round(
        cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    if manual_select_roi:
        hasFrame, frame = cap.read()
        if not hasFrame:
            sys.exit("not able to read frame !")
        else:
            # Select ROI 
            xi, yi, bboxWidth, bboxHeight = cv.selectROI("select the area", frame)
            x0, x1 = xi, xi + bboxWidth
            y0, y1 = yi, yi + bboxHeight
            template = frame[y0:y1, x0:x1]

            assert y0-margin > 0 and x0-margin > 0 and y1+margin<frame.shape[0] and x1+margin<frame.shape[1], "outside frame"
            y0, y1, x0, x1 = y0-margin, y1+margin, x0-margin, x1+margin # for searchBox

            is_roi_inside = True    

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Done processing !")
            break

        # Prepare the frame for detection
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))

        multiTracker = updateTrackersAndDetections(frame, outs, multiTracker)

        if manual_select_roi  and is_roi_inside:
            searchBox = frame[y0:y1, x0:x1]  
            max_loc = match(baseImg=searchBox, templateImg=template)
            print(f"{max_loc=}")
            xShift, yShift = max_loc[0] - margin, max_loc[1] - margin
            x0, x1, y0, y1 =  x0 + xShift, x1 + xShift, y0 + yShift, y1 + yShift
            print(x0, x1, y0, y1)
            if x0 > 0 and y0 > 0 and x1 < frame.shape[1] and y1 < frame.shape[0]:

                left, top = x0 + margin , y0 + margin
                cv.rectangle(frame, (left, top), (left + bboxWidth, top + bboxHeight), (0,0,255), 2, 1)
            else:
                is_roi_inside = False

        # Display the frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Allow breaking the loop with 'q'
            break

        # Write the frame with the detection boxes

        vid_writer.write(frame.astype(np.uint8))

    cv.destroyAllWindows()
